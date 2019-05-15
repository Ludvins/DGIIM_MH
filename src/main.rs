use k_nn::algorithms::p2::*;
use k_nn::types::data::*;
use rand::prelude::*;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut seed = 2;
    if args.len() > 1 {
        seed = args[1].parse::<u64>().unwrap();
    }
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);

    let do_texture = true;
    let do_colpos = true;
    let do_iono = true;

    println!("# Current Results.");
    if do_texture {
        println!("## Results for Texture.\n");
        if let Err(err) = run_p2::<Texture>(String::from("data/texture.csv"), 40, 5, &mut rng) {
            println!("Error running Texture: {}", err);
            std::process::exit(1);
        }
    }

    if do_colpos {
        println!("## Results for Colposcopy.\n");
        if let Err(err) = run_p2::<Colposcopy>(String::from("data/colposcopy.csv"), 62, 5, &mut rng)
        {
            println!("Error running Colposcopy: {}", err);
            std::process::exit(1);
        }
    }

    if do_iono {
        println!("## Results for Ionosphere.\n");
        if let Err(err) = run_p2::<Ionosphere>(String::from("data/ionosphere.csv"), 34, 5, &mut rng)
        {
            println!("Error running Ionosphere: {}", err);
            std::process::exit(1);
        }
    }
}
