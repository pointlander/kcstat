// Copyright 2017 The KCStat Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math/rand"

	ga "github.com/pointlander/go-galib"
)

type GACDF2Initializer struct{}

func (i *GACDF2Initializer) InitPop(first ga.GAGenome, popsize int) (pop []ga.GAGenome) {
	pop = make([]ga.GAGenome, popsize)
	pop[0] = first.Copy().(*ga.GAFloat32Genome)
	for x := 1; x < popsize; x++ {
		genome := first.Copy().(*ga.GAFloat32Genome)
		pop[x] = genome
		for i := 0; i < 256; i++ {
			offset := i * 256
			for j := 0; j < 256; j++ {
				genome.Gene[offset+j] += rand.Float32() * .1
				if genome.Gene[offset+j] > 1 {
					genome.Gene[offset+j] = 1
				}
			}
		}
	}
	return pop
}

func (i *GACDF2Initializer) String() string { return "CDF2Initializer" }

type GACDFInitializer struct{}

func (i *GACDFInitializer) InitPop(first ga.GAGenome, popsize int) (pop []ga.GAGenome) {
	pop = make([]ga.GAGenome, popsize)
	for x := 0; x < popsize; x++ {
		genome := first.Copy().(*ga.GAFloat32Genome)
		pop[x] = genome
		for i := 0; i < 256; i++ {
			offset := i * 256
			for j := 0; j < 256; j++ {
				genome.Gene[offset+j] = rand.Float32() * .1
				if i == j {
					genome.Gene[offset+j] = 1
				}
			}
		}
	}
	return pop
}

func (i *GACDFInitializer) String() string { return "CDFInitializer" }

type GABoundedGaussianMutator struct {
	StdDev float64
	Mean   float64
}

func NewGABoundedGaussianMutator(stddev float64, mean float64) *GABoundedGaussianMutator {
	if stddev == 0 {
		return nil
	}
	return &GABoundedGaussianMutator{StdDev: stddev, Mean: mean}
}

func (m GABoundedGaussianMutator) Mutate(a ga.GAGenome) ga.GAGenome {
	switch a := a.(type) {
	case *ga.GAFloatGenome:
		n := a.Copy().(*ga.GAFloatGenome)
		l := a.Len()
		s := rand.Intn(l)
		n.Gene[s] += rand.NormFloat64()*m.StdDev + m.Mean
		if n.Gene[s] < n.Min {
			n.Gene[s] = n.Min
		} else if n.Gene[s] > n.Max {
			n.Gene[s] = n.Max
		}
		return n
	case *ga.GAFloat32Genome:
		n := a.Copy().(*ga.GAFloat32Genome)
		l := a.Len()
		s := rand.Intn(l)
		n.Gene[s] += float32(rand.NormFloat64()*m.StdDev + m.Mean)
		if n.Gene[s] < n.Min {
			n.Gene[s] = n.Min
		} else if n.Gene[s] > n.Max {
			n.Gene[s] = n.Max
		}
		return n
	}

	return nil
}

func (m GABoundedGaussianMutator) String() string { return "GABoundedGaussianMutator" }
