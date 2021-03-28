package bayesfactor

import (
	. "github.com/ljcolling/go-distributions"
	"math"
)

func Bayesfactor(likelihood LikelihoodDef, altprior PriorDef, nullprior PriorDef) (float64, error) {

	var data Likelihood
	var alt Prior
	var null Prior
	switch likelihood.Name {
	case "noncentral_d":
		d := likelihood.Params[0]
		df := likelihood.Params[1]
		fun := Noncentral_d_likelihood(d, df)
		data.function = fun
		data.name = "noncentral_d"

	case "normal":
		mean := likelihood.Params[0]
		sd := likelihood.Params[1]
		fun := Normal_likelihood(mean, sd)
		data.function = fun
		data.name = "normal"

	case "binomial":
		successes := likelihood.Params[0]
		trials := likelihood.Params[1]
		fun := Binomial_likelihood(successes, trials)
		data.function = fun
		data.name = "binomial"

	case "noncentral_t":
		t := likelihood.Params[0]
		df := likelihood.Params[1]
		fun := Noncentral_t_likelihood(t, df)
		data.function = fun
		data.name = "noncentral_t"

	case "student_t":
		mean := likelihood.Params[0]
		sd := likelihood.Params[1]
		df := likelihood.Params[2]
		fun := Student_t_likelihood(mean, sd, df)
		data.function = fun
		data.name = "student_t"
	}

	switch nullprior.Name {
	case "cauchy":
		location := nullprior.Params[0]
		scale := nullprior.Params[1]
		min := nullprior.Params[2]
		max := nullprior.Params[3]
		null = Cauchy_prior(location, scale, min, max)

	case "normal":
		mean := nullprior.Params[0]
		sd := nullprior.Params[1]
		min := nullprior.Params[2]
		max := nullprior.Params[3]
		null = Normal_prior(mean, sd, min, max)

	case "beta":
		alpha := nullprior.Params[0]
		beta := nullprior.Params[1]
		null = Beta_prior(alpha, beta, 0, 1)

	case "uniform":
		alpha := nullprior.Params[0]
		beta := nullprior.Params[1]
		null = Uniform_prior(alpha, beta)

	case "student_t":
		mean := nullprior.Params[0]
		sd := nullprior.Params[1]
		df := nullprior.Params[2]
		min := nullprior.Params[3]
		max := nullprior.Params[4]
		null = Student_t_prior(mean, sd, df, min, max)
	case "point":
		point := nullprior.Params[0]
		null = point_prior(point)
	}

	switch altprior.Name {
	case "cauchy":
		location := altprior.Params[0]
		scale := altprior.Params[1]
		min := altprior.Params[2]
		max := altprior.Params[3]
		alt = Cauchy_prior(location, scale, min, max)

	case "normal":
		mean := altprior.Params[0]
		sd := altprior.Params[1]
		min := altprior.Params[2]
		max := altprior.Params[3]
		alt = Normal_prior(mean, sd, min, max)

	case "beta":
		alpha := altprior.Params[0]
		beta := altprior.Params[1]
		alt = Beta_prior(alpha, beta, 0, 1)

	case "uniform":
		alpha := altprior.Params[0]
		beta := altprior.Params[1]
		alt = Uniform_prior(alpha, beta)

	case "student_t":
		mean := altprior.Params[0]
		sd := altprior.Params[1]
		df := altprior.Params[2]
		min := altprior.Params[3]
		max := altprior.Params[4]
		alt = Student_t_prior(mean, sd, df, min, max)
	}

	alt_model := pp(data, alt)
	nul_model := pp(data, null)
	bf := alt_model.auc / nul_model.auc
	return bf, nil
}

// Types
type LikelihoodDef struct {
	Name   string
	Params []float64
}

type PriorDef struct {
	Name   string
	Params []float64
}

// Output types
// Predctive type
type Predictive struct {
	function   func(x float64) float64
	auc        float64
	likelihood func(x float64) float64
	prior      func(x float64) float64
}

// Prior type
type Prior struct {
	Function func(x float64) float64
	name     string
	point    float64 // this is only used for the point prior because floating point :(
}

// Likelihood type
type Likelihood struct {
	function func(x float64) float64
	name     string
}

// Helper functions
func inrange(x float64, min float64, max float64) float64 {
	if x >= min && x <= max {
		return 1
	} else {
		return 0
	}
}

func mult(likelihood func(x float64) float64, prior func(x float64) float64) func(x float64) float64 {
	return func(x float64) float64 {
		return likelihood(x) * prior(x)
	}
}

func pp(likelihood Likelihood, prior Prior) Predictive {

	var prod func(x float64) float64
	likelihood_function := likelihood.function
	prod = mult(likelihood_function, prior.Function)
	var pred Predictive
	pred.function = prod
	switch prior.name {
	case "point":
		pred.auc = likelihood_function(prior.point)
		goto END
	}
	switch likelihood.name {
	case "binomial":
		pred.auc = Integrate(prod, 0, 1)
	default:
		pred.auc = Integrate(prod, math.Inf(-1), math.Inf(1))
	}

END:
	pred.likelihood = likelihood.function
	pred.prior = prior.Function

	return pred

}

// normal likelihood
func Normal_likelihood(mean float64, sd float64) func(x float64) float64 {
	return func(x float64) float64 {
		return Dnorm(x, mean, sd)
	}
}

// student-t likelihood
func Student_t_likelihood(mean float64, sd float64, df float64) func(x float64) float64 {
	return func(x float64) float64 {
		return Scaled_shifted_t(x, mean, sd, df)
	}
}

// noncentral t likehood
func Noncentral_t_likelihood(t float64, df float64) func(x float64) float64 {
	return func(x float64) float64 {
		return Dt(t, df, x)
	}
}

// noncentral d likelihood
func Noncentral_d_likelihood(d float64, df float64) func(x float64) float64 {
	return func(x float64) float64 {
		return Dt(d*math.Sqrt(df+1), df, math.Sqrt(df+1)*x)
	}
}

// binomial likelihood
func Binomial_likelihood(successes float64, trials float64) func(x float64) float64 {
	return func(x float64) float64 {
		return Dbinom(successes, trials, x)
	}
}

// normal prior
func Normal_prior(mean float64, sd float64, min float64, max float64) Prior {

	// If max and max are +/-Inf then set K to 1
	// otherwise, integrate and normalize
	if min == math.Inf(-1) && max == math.Inf(1) {

		var prior Prior
		prior.point = 0
		prior.Function = func(x float64) float64 {
			return Dnorm(x, mean, sd)
		}
		prior.name = "normal"
		return prior
	} else if (min == 0.0 && max == math.Inf(1)) || (min == math.Inf(-1) && max == 0.0) {
		k := 2.0
		var prior Prior
		prior.Function = func(x float64) float64 {
			return (Dnorm(x, mean, sd) * inrange(x, min, max)) * k
		}
		prior.name = "normal"
		return prior
	} else {
		normal := func(x float64) float64 {
			return Dnorm(x, mean, sd) * inrange(x, min, max)
		}
		auc := Integrate(normal, math.Inf(-1), math.Inf(1))
		k := 1 / auc
		var prior Prior
		prior.Function = func(x float64) float64 {
			return (Dnorm(x, mean, sd) * inrange(x, min, max)) * k
		}
		prior.name = "normal"
		return prior
	}
}

// student t prior
func Student_t_prior(mean float64, sd float64, df float64, min float64, max float64) Prior {

	// If max and max are +/-Inf then set K to 1
	// otherwise, integrate and normalize
	if min == math.Inf(-1) && max == math.Inf(1) {

		var prior Prior
		prior.point = 0
		prior.Function = func(x float64) float64 {
			return Scaled_shifted_t(x, mean, sd, df)
		}
		prior.name = "student_t"
		return prior
	} else if (min == 0.0 && max == math.Inf(1)) || (min == math.Inf(-1) && max == 0.0) {
		k := 2.0
		var prior Prior
		prior.Function = func(x float64) float64 {
			return (Scaled_shifted_t(x, mean, sd, df) * inrange(x, min, max)) * k
		}
		prior.name = "student_t"
		return prior
	} else {
		normal := func(x float64) float64 {
			return Scaled_shifted_t(x, mean, sd, df) * inrange(x, min, max)
		}
		auc := Integrate(normal, math.Inf(-1), math.Inf(1))
		k := 1 / auc
		var prior Prior
		prior.Function = func(x float64) float64 {
			return (Scaled_shifted_t(x, mean, sd, df) * inrange(x, min, max)) * k
		}
		prior.name = "student_t"
		return prior
	}
}

// cauchy prior
func Cauchy_prior(location float64, scale float64, min float64, max float64) Prior {

	// If max and max are +/-Inf then set K to 1
	// otherwise, integrate and normalize
	if min == math.Inf(-1) && max == math.Inf(1) {

		var prior Prior
		prior.point = 0
		prior.Function = func(x float64) float64 {
			return Dcauchy(x, location, scale)
		}
		prior.name = "cauchy"
		return prior
	} else if (min == 0.0 && max == math.Inf(1)) || (min == math.Inf(-1) && max == 0.0) {
		k := 2.0
		var prior Prior
		prior.Function = func(x float64) float64 {
			return (Dcauchy(x, location, scale) * inrange(x, min, max)) * k
		}
		prior.name = "cauchy"
		return prior
	} else {
		cauchy := func(x float64) float64 {
			return Dcauchy(x, location, scale) * inrange(x, min, max)
		}
		auc := Integrate(cauchy, math.Inf(-1), math.Inf(1))
		k := 1 / auc
		var prior Prior
		prior.Function = func(x float64) float64 {
			return (Dcauchy(x, location, scale) * inrange(x, min, max)) * k
		}
		prior.name = "cauchy"
		return prior
	}
}

// beta prior
func Beta_prior(alpha float64, beta float64, min float64, max float64) Prior {

	var prior Prior
	prior.point = 0
	prior.Function = func(x float64) float64 {
		return Dbeta(x, alpha, beta) * inrange(x, min, max)
	}
	prior.name = "beta"
	return prior
}

// point prior
func point_prior(point float64) Prior {

	var prior Prior
	prior.Function = func(x float64) float64 {
		if x == point {
			return 1.0
		} else {
			return 0
		}
	}
	prior.name = "point"
	prior.point = point
	return prior
}

// uniform prior
func Uniform_prior(alpha float64, beta float64) Prior {

	var prior Prior
	prior.point = 0
	prior.Function = func(x float64) float64 {
		return Dunif(x, alpha, beta)
	}
	prior.name = "uniform"
	return prior
}
