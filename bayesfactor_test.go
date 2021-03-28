package bayesfactor

import (
	"github.com/google/go-cmp/cmp"
	"math"
	"testing"
)

func TestBayesfactor(t *testing.T) {

	const tolerance = .0001
	opt := cmp.Comparer(func(x, y float64) bool {
		diff := math.Abs(x - y)
		mean := math.Abs(x+y) / 2.0
		return (diff / mean) < tolerance
	})

	var likelihood LikelihoodDef // likelihood
	var altprior PriorDef        // alternative prior
	var nullprior PriorDef       // null prior
	//// likelihood
	likelihood.Name = "noncentral_d"
	likelihood.Params = []float64{2.03 / math.Sqrt(80), 79.0}
	//// alt prior
	altprior.Name = "cauchy"
	altprior.Params = []float64{0, 1, math.Inf(-1), math.Inf(1)}
	//// null prior
	nullprior.Name = "point"
	nullprior.Params = []float64{0.0}
	bf, _ := Bayesfactor(likelihood, altprior, nullprior)
	got := bf
	want := 1 / 1.557447
	if !cmp.Equal(got, want, opt) {
		t.Fatalf("got %v, wanted %v", got, want)
	}

	likelihood.Name = "noncentral_t"
	likelihood.Params = []float64{2.03, 79.0}
	//// alt prior
	altprior.Name = "cauchy"
	altprior.Params = []float64{0, 1 * math.Sqrt(80), math.Inf(-1), math.Inf(1)}
	//// null prior
	nullprior.Name = "point"
	nullprior.Params = []float64{0.0}
	bf, _ = Bayesfactor(likelihood, altprior, nullprior)
	got = bf
	want = 1 / 1.557447
	if !cmp.Equal(got, want, opt) {
		t.Fatalf("got %v, wanted %v", got, want)
	}

	likelihood.Name = "normal"
	likelihood.Params = []float64{5.5, 32.35}
	//// alt prior
	altprior.Name = "normal"
	altprior.Params = []float64{0, 13.3, 0, math.Inf(1)}
	//// null prior
	nullprior.Name = "point"
	nullprior.Params = []float64{0.0}
	bf, _ = Bayesfactor(likelihood, altprior, nullprior)
	got = bf
	want = 0.9745934
	if !cmp.Equal(got, want, opt) {
		t.Fatalf("got %v, wanted %v", got, want)
	}

	likelihood.Name = "normal"
	likelihood.Params = []float64{5, 10}
	//// alt prior
	altprior.Name = "uniform"
	altprior.Params = []float64{0, 20.0}
	//// null prior
	nullprior.Name = "point"
	nullprior.Params = []float64{0.0}
	bf, _ = Bayesfactor(likelihood, altprior, nullprior)
	got = bf
	want = 0.887226
	if !cmp.Equal(got, want, opt) {
		t.Fatalf("got %v, wanted %v", got, want)
	}

	likelihood.Name = "binomial"
	likelihood.Params = []float64{8, 11}
	//// alt prior
	altprior.Name = "beta"
	altprior.Params = []float64{2.5, 1}
	//// null prior
	nullprior.Name = "point"
	nullprior.Params = []float64{0.5}
	bf, _ = Bayesfactor(likelihood, altprior, nullprior)
	got = bf
	want = 1 / 0.6632996
	if !cmp.Equal(got, want, opt) {
		t.Fatalf("got %v, wanted %v", got, want)
	}

	likelihood.Name = "binomial"
	likelihood.Params = []float64{2, 10}
	//// alt prior
	altprior.Name = "normal"
	altprior.Params = []float64{0, 1, 0, 1}
	//// null prior
	nullprior.Name = "point"
	nullprior.Params = []float64{0.5}
	bf, _ = Bayesfactor(likelihood, altprior, nullprior)
	got = bf
	want = 2.327971
	if !cmp.Equal(got, want, opt) {
		t.Fatalf("got %v, wanted %v", got, want)
	}

	likelihood.Name = "student_t"
	likelihood.Params = []float64{5.47, 32.2, 119}
	//// alt prior
	altprior.Name = "student_t"
	altprior.Params = []float64{13.3, 4.93, 72, math.Inf(-1), math.Inf(1)}
	//// null prior
	nullprior.Name = "point"
	nullprior.Params = []float64{0}
	bf, _ = Bayesfactor(likelihood, altprior, nullprior)
	got = bf
	want = 0.9738
	if !cmp.Equal(got, want, opt) {
		t.Fatalf("got %v, wanted %v", got, want)
	}

}

func BenchmarkPlots(b *testing.B) {

	mean := 0.0
	sd := 1.0
	min := mean - 4*sd // min range of plot
	max := mean + 4*sd // max range  of plot
	result := []interface{}{}

	likelihood_function := Normal_likelihood(mean, sd)

	step := (max - min) / 100
	x := min
	for i := 0; i < 101; i++ {
		y := likelihood_function(x)
		res := map[string]interface{}{"x": x, "y": y}
		result = append(result, res)
		x += step
	}

}

func BenchmarkDefault(b *testing.B) {

	var likelihood LikelihoodDef // likelihood
	var altprior PriorDef        // alternative prior
	var nullprior PriorDef       // null prior
	//// likelihood
	likelihood.Name = "noncentral_d"
	likelihood.Params = []float64{2.03 / math.Sqrt(80), 79.0}
	//// alt prior
	altprior.Name = "cauchy"
	altprior.Params = []float64{0, 1, math.Inf(-1), math.Inf(1)}
	//// null prior
	nullprior.Name = "point"
	nullprior.Params = []float64{0.0}
	Bayesfactor(likelihood, altprior, nullprior)

}
