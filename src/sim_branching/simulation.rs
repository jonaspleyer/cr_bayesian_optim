use approxim::AbsDiffEq;
use cellular_raza::prelude as cr;
use nalgebra::Vector2;
use num::Zero;
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

pub type ReactionVector = nalgebra::DVector<f64>;

macro_rules! opt (
    ($($ti:tt)*) => {
        short_default::default! {
            #[pyclass(get_all, set_all)]
            #[derive(Clone, Debug, AbsDiffEq, PartialEq)]
            #[approx(epsilon_type = f64)]
            pub struct $($ti)*
        }
    }
);

opt! { BacterialParameters {
    #[approx(equal)]
    pub n_bacteria_initial: u32 = 5,
    pub cell_radius: f64 = 6.0,
    pub division_threshold: f64 = 2.0,
    pub potential_stiffness: f64 = 0.15,
    pub potential_strength: f64 = 2.0,
    pub damping_constant: f64 = 1.0,
    pub uptake_rate: f64 = 1.0,
    pub growth_rate: f64 = 13.0,
}}

opt! { DomainParameters {
    /// Overall size of the domain
    pub domain_size: f64 = 3000.0,
    pub voxel_size: f64 = 30.0,
    /// Size of the square for initlal placement of bacteria
    pub domain_starting_size: f64 = 100.0,
    /// Discretization of the diffusion process
    pub reactions_dx: f64 = 20.0,
    pub diffusion_constant: f64 = 80.0,
    pub initial_concentration: f64 = 10.0,
}}

opt! { TimeParameters {
    pub dt: f64 = 0.1,
    pub t_max: f64 = 2000.0,
    #[approx(equal)]
    pub save_interval: usize = 200,
}}

#[pyclass]
#[derive(Clone)]
pub struct Options {
    pub bacteria: Py<BacterialParameters>,
    pub domain: Py<DomainParameters>,
    pub time: Py<TimeParameters>,
    pub n_threads: usize,
}

impl PartialEq for Options {
    fn eq(&self, other: &Self) -> bool {
        Python::with_gil(|py| {
            self.bacteria.borrow(py).eq(&other.bacteria.borrow(py))
                && self.domain.borrow(py).eq(&other.domain.borrow(py))
                && self.time.borrow(py).eq(&other.time.borrow(py))
                && self.n_threads.eq(&other.n_threads)
        })
    }
}

impl approxim::AbsDiffEq for Options {
    type Epsilon = f64;
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Python::with_gil(|py| {
            self.bacteria
                .borrow(py)
                .abs_diff_eq(&other.bacteria.borrow(py), epsilon)
                && self
                    .domain
                    .borrow(py)
                    .abs_diff_eq(&other.domain.borrow(py), epsilon)
                && self
                    .time
                    .borrow(py)
                    .abs_diff_eq(&other.time.borrow(py), epsilon)
                && self.n_threads.abs_diff_eq(&other.n_threads, 0)
        })
    }

    fn default_epsilon() -> Self::Epsilon {
        <f64 as AbsDiffEq>::default_epsilon()
    }
}

#[pyfunction]
pub fn run_sim_branching(options: Options, py: Python) -> Result<(), cr::SimulationError> {
    use cr::*;

    let BacterialParameters {
        n_bacteria_initial,
        cell_radius,
        division_threshold,
        potential_stiffness,
        potential_strength,
        damping_constant,
        uptake_rate,
        growth_rate,
    } = *options.bacteria.borrow(py);
    let DomainParameters {
        domain_size,
        voxel_size,
        domain_starting_size,
        reactions_dx,
        diffusion_constant,
        initial_concentration,
    } = *options.domain.borrow(py);
    let TimeParameters {
        dt,
        t_max,
        save_interval,
    } = *options.time.borrow(py);
    let n_threads = options.n_threads;

    let ds = domain_size / 2.0;
    let dx = domain_starting_size / 2.0;

    // Fix random seed
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    let cells = (0..n_bacteria_initial)
        .map(|_| {
            let x = rng.random_range(ds - dx..ds + dx);
            let y = rng.random_range(ds - dx..ds + dx);

            let pos = Vector2::from([x, y]);
            super::MyAgent {
                mechanics: NewtonDamped2D {
                    pos,
                    vel: Vector2::zero(),
                    damping_constant,
                    mass: 1.0,
                },
                interaction: MorsePotential {
                    radius: cell_radius,
                    potential_stiffness,
                    cutoff: 2.0 * division_threshold * cell_radius,
                    strength: potential_strength,
                },
                uptake_rate,
                division_radius: division_threshold * cell_radius,
                growth_rate,
            }
        })
        .collect::<Vec<_>>();

    let cond = dt - 0.5 * reactions_dx / diffusion_constant;
    if cond >= 0.0 {
        println!(
            "❗❗❗WARNING❗❗❗\n\
            The stability condition \
            dt <= 0.5 dx^2/D for the integration \
            method is not satisfied. This can \
            lead to solving errors and inaccurate \
            results."
        );
    }

    if voxel_size < division_threshold * cell_radius {
        println!(
            "❗❗❗WARNING❗❗❗\n\
            The voxel_size {voxel_size} has been chosen \
            smaller than the length of the interaction {}. This \
            will probably yield incorrect results.",
            division_threshold * cell_radius,
        );
    }

    let domain = CartesianDiffusion2D {
        domain: CartesianCuboid::from_boundaries_and_interaction_range(
            [0.0; 2],
            [domain_size, domain_size],
            voxel_size,
        )?,
        reactions_dx: [reactions_dx; 2].into(),
        diffusion_constant,
        initial_value: ReactionVector::from(vec![initial_concentration]),
    };

    let storage = StorageBuilder::new().priority([StorageOption::SerdeJson]);
    let time = FixedStepsize::from_partial_save_freq(0.0, dt, t_max, save_interval)?;
    let settings = Settings {
        n_threads: n_threads.try_into().unwrap(),
        time,
        storage,
        show_progressbar: true,
    };

    let _storager = run_simulation!(
        agents: cells,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, Interaction, ReactionsExtra, Cycle],
        parallelizer: Rayon,
        zero_reactions_default: |_| nalgebra::DVector::zeros(1),
    )?;
    Ok(())
}
