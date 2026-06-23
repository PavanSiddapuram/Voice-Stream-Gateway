// Package audio handles PCM audio preprocessing: linear-interpolation resampling
// and energy-based Voice Activity Detection (VAD).
//
// In Python these loops run under the GIL and become a serialisation point at
// high concurrency. In Go each goroutine runs them independently with zero
// contention, keeping processing time flat regardless of session count.
package audio

import "math"

// Resample converts raw int16 PCM from srcRate to dstRate using linear
// interpolation. Input is little-endian int16 bytes; output is the same format.
func Resample(src []byte, srcRate, dstRate int) []byte {
	if srcRate == dstRate {
		return src
	}
	samples := make([]float64, len(src)/2)
	for i := range samples {
		lo := src[i*2]
		hi := src[i*2+1]
		samples[i] = float64(int16(uint16(lo) | uint16(hi)<<8))
	}

	ratio := float64(srcRate) / float64(dstRate)
	outLen := int(math.Round(float64(len(samples)) / ratio))
	out := make([]byte, outLen*2)

	for i := 0; i < outLen; i++ {
		pos := float64(i) * ratio
		lo := int(pos)
		hi := lo + 1
		if hi >= len(samples) {
			hi = len(samples) - 1
		}
		frac := pos - float64(lo)
		val := int16(samples[lo]*(1-frac) + samples[hi]*frac)
		out[i*2] = byte(val)
		out[i*2+1] = byte(val >> 8)
	}
	return out
}

// VADResult is the outcome of a single VAD frame analysis.
type VADResult struct {
	IsSpeech     bool
	EnergyRMS    float64
	SpeechFrames int // consecutive speech frames seen so far
}

// VAD is a stateful energy-based Voice Activity Detector.
// It uses a simple RMS energy threshold with a short hangover window so brief
// silences within speech are not mis-labelled as pauses.
type VAD struct {
	ThresholdDB    float64 // energy threshold (dB), e.g. -40.0
	HangoverFrames int     // frames to stay "speech" after energy drops
	hangover       int
	SpeechFrames   int
}

func NewVAD() *VAD {
	return &VAD{
		ThresholdDB:    -40.0,
		HangoverFrames: 8,
	}
}

// Process accepts a single frame of int16 PCM bytes and returns a VADResult.
func (v *VAD) Process(frame []byte) VADResult {
	rms := rmsEnergy(frame)
	db := 20 * math.Log10(rms+1e-9)
	isSpeech := db > v.ThresholdDB

	if isSpeech {
		v.hangover = v.HangoverFrames
		v.SpeechFrames++
	} else if v.hangover > 0 {
		v.hangover--
		isSpeech = true // hangover: keep speech flag alive briefly
	} else {
		v.SpeechFrames = 0
	}

	return VADResult{
		IsSpeech:     isSpeech,
		EnergyRMS:    rms,
		SpeechFrames: v.SpeechFrames,
	}
}

func rmsEnergy(frame []byte) float64 {
	if len(frame) < 2 {
		return 0
	}
	var sum float64
	n := len(frame) / 2
	for i := 0; i < n; i++ {
		lo := frame[i*2]
		hi := frame[i*2+1]
		s := float64(int16(uint16(lo) | uint16(hi)<<8))
		sum += s * s
	}
	return math.Sqrt(sum / float64(n))
}
