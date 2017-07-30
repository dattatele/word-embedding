// Copyright © 2017 Makoto Ito
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package glove

import (
	"bufio"
	"io"
	"runtime"
	"strings"

	"github.com/chewxy/lingo/corpus"
	"github.com/pkg/errors"
	"github.com/ynqa/word-embedding/models"
	"github.com/chewxy/gorgonia/tensor"
)

var procs int

func init() {
	procs = runtime.GOMAXPROCS(-1)
}

type GloVe struct {
	*corpus.Corpus
	*models.Common
	PairMap
	FactorMatrix
	Epoch int
	Alpha float64
}

func NewGloVe(config models.Common, epoch int, alpha float64) *GloVe {
	return &GloVe{
		Corpus: corpus.New(),
		Common: config,
		PairMap: new(PairMap),
		Alpha: alpha,
		Epoch: epoch,
	}
}

type FactorMatrix struct {
	F tensor.Tensor
	Q tensor.Tensor
}

func NewFactorMatrix(vocab, dim int) *FactorMatrix {
	return &FactorMatrix{
		F: tensor.New(tensor.WithShape(vocab * dim)),
		Q: tensor.New(tensor.WithShape(vocab * dim)),
	}
}

func (g *GloVe) update(wordIDs []int) {
	for index := range wordIDs {
		for i := 0; i < g.Common.Window; i++ {
			if index + i >= len(wordIDs){
				break
			}
			g.PairMap.Update(wordIDs[index], wordIDs[index+i])
		}
	}
	return
}

func (g *GloVe) Preprocess(f io.ReadSeeker) (io.ReadCloser, error) {
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := s.Text()
		line = strings.ToLower(line)
		words := strings.Fields(line)
		wordIDs := g.toIDs(words)
		g.update(wordIDs)
		if err := s.Err(); err != nil && err != io.EOF {
			return errors.Wrap(err, "Unable to complete scanning.")
		}
	}
	g.FactorMatrix = NewFactorMatrix(g.Size(), g.Dimension)
	return
}

func (g *GloVe) Train(f io.ReadCloser) error {
	if g.FactorMatrix == nil {
		return errors.Errorf("No initialize model parameters")
	}

	f.Close()
	for i := 0; i < g.Epoch; i++ {
		for p, f := range g.PairMap {
			g.train(p, f)
		}
	}
	return nil
}

func (g *GloVe) train(p Pair, f int) {}

func (g *GloVe) toIDs(words []string) []int {
	retVal := make([]int, len(words))
	for i, w := range words {
		retVal[i], _ = g.Id(w)
	}
	return retVal
}

func (g *GloVe) toWords(wordIDs []int) []string {
	retVal := make([]string, len(wordIDs))
	for i, w := range wordIDs {
		retVal[i], _ = g.Word(w)
	}
	return retVal
}