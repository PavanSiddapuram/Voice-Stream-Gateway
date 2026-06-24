// Package ring implements a consistent hash ring for KV-cache-aware sticky
// session routing. Each session_id always maps to the same GPU worker node,
// maximising prefix-cache hit rate across reconnects.
package ring

import (
	"crypto/md5"
	"encoding/binary"
	"fmt"
	"sort"
	"sync"
)

type HashRing struct {
	mu       sync.RWMutex
	replicas int
	points   []uint32
	nodeMap  map[uint32]string
}

func New(nodes []string, replicas int) *HashRing {
	r := &HashRing{
		replicas: replicas,
		nodeMap:  make(map[uint32]string),
	}
	for _, n := range nodes {
		r.AddNode(n)
	}
	return r
}

func (r *HashRing) hash(key string) uint32 {
	h := md5.Sum([]byte(key))
	return binary.BigEndian.Uint32(h[:4])
}

func (r *HashRing) AddNode(node string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for i := 0; i < r.replicas; i++ {
		v := r.hash(fmt.Sprintf("%s#%d", node, i))
		r.points = append(r.points, v)
		r.nodeMap[v] = node
	}
	sort.Slice(r.points, func(i, j int) bool { return r.points[i] < r.points[j] })
}

func (r *HashRing) RemoveNode(node string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for i := 0; i < r.replicas; i++ {
		v := r.hash(fmt.Sprintf("%s#%d", node, i))
		delete(r.nodeMap, v)
	}
	// Rebuild sorted slice without removed node's virtual points.
	pts := r.points[:0]
	for _, p := range r.points {
		if _, ok := r.nodeMap[p]; ok {
			pts = append(pts, p)
		}
	}
	r.points = pts
}

// GetNode returns the worker node that owns key — O(log n).
func (r *HashRing) GetNode(key string) string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if len(r.points) == 0 {
		return ""
	}
	v := r.hash(key)
	idx := sort.Search(len(r.points), func(i int) bool { return r.points[i] >= v })
	if idx == len(r.points) {
		idx = 0
	}
	return r.nodeMap[r.points[idx]]
}
