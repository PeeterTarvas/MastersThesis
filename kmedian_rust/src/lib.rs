use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{Array2, Array1, s};

fn pairwise_l1(x: &Array2<f64>, centers: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let k = centers.nrows();
    let d = x.ncols();
    let mut dist = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            let mut s = 0.0;
            for f in 0..d {
                s += (x[[i, f]] - centers[[j, f]]).abs();
            }
            dist[[i, j]] = s;
        }
    }
    dist
}

/// given the full distance matrix D (n×k), derive labels, per-point min distances,
/// and the weighted total cost.
fn labels_and_cost(
    d: &Array2<f64>,
    weights: &Array1<f64>,
) -> (Array1<i32>, Array1<f64>, f64) {
    let n = d.nrows();
    let k = d.ncols();
    let mut labels = Array1::<i32>::zeros(n);
    let mut min_dists = Array1::<f64>::zeros(n);
    let mut cost = 0.0;

    for i in 0..n {
        let mut best_dist = f64::MAX;
        let mut best_j = 0i32;
        for j in 0..k {
            let v = d[[i, j]];
            if v < best_dist {
                best_dist = v;
                best_j = j as i32;
            }
        }
        labels[i] = best_j;
        min_dists[i] = best_dist;
        cost += weights[i] * best_dist;
    }
    (labels, min_dists, cost)
}

#[pyfunction]
fn local_search_kmedian<'py>(
    py: Python<'py>,
    x_py: PyReadonlyArray2<f64>,
    k: usize,
    weights_py: PyReadonlyArray1<f64>,
    init_centers_py: PyReadonlyArray2<f64>,
    max_iter: usize,
) -> PyResult<(&'py PyArray2<f64>, &'py PyArray1<i32>, f64)> {

    let x = x_py.as_array().to_owned();
    let weights = weights_py.as_array().to_owned();
    let mut centers = init_centers_py.as_array().to_owned();

    let n = x.nrows();
    let d = x.ncols();

    // full n×k distance matrix — reused across iterations
    let mut big_d = pairwise_l1(&x, &centers);
    let (mut labels, mut _min_dists, mut cost) = labels_and_cost(&big_d, &weights);

    for _ in 0..max_iter {
        let mut best_gain = 0.0f64;
        let mut best_swap: Option<(usize, usize)> = None;

        for ci in 0..k {
            // for each point, find the best distance ignoring center `ci`.
            // this is the minimum over all columns except column `ci`.
            let mut min_dists_without_ci = Array1::<f64>::from_elem(n, f64::MAX);
            for i in 0..n {
                let mut best = f64::MAX;
                for j in 0..k {
                    if j == ci {
                        continue;
                    }
                    let v = big_d[[i, j]];
                    if v < best {
                        best = v;
                    }
                }
                min_dists_without_ci[i] = best;
            }

            // try swapping center `ci` with each data point `xi`
            for xi in 0..n {
                // compute distance from every point to the candidate xi
                let mut trial_cost = 0.0f64;
                for i in 0..n {
                    let mut dist_to_cand = 0.0;
                    for f in 0..d {
                        dist_to_cand += (x[[i, f]] - x[[xi, f]]).abs();
                    }
                    // each point goes to whichever is closer
                    let new_dist = min_dists_without_ci[i].min(dist_to_cand);
                    trial_cost += weights[i] * new_dist;
                }

                let gain = cost - trial_cost;
                if gain > best_gain {
                    best_gain = gain;
                    best_swap = Some((ci, xi));
                }
            }
        }

        if let Some((ci, xi)) = best_swap {
            // apply the swap
            let new_center = x.slice(s![xi, ..]).to_owned();
            centers.row_mut(ci).assign(&new_center);

            // update only the column `ci` of big_d
            for i in 0..n {
                let mut dist = 0.0;
                for f in 0..d {
                    dist += (x[[i, f]] - centers[[ci, f]]).abs();
                }
                big_d[[i, ci]] = dist;
            }

            let result = labels_and_cost(&big_d, &weights);
            labels = result.0;
            _min_dists = result.1;
            cost = result.2;
        } else {
            break; // local optimum reached
        }
    }

    Ok((centers.into_pyarray(py), labels.into_pyarray(py), cost))
}

#[pymodule]
fn fast_kmedian(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(local_search_kmedian, m)?)?;
    Ok(())
}