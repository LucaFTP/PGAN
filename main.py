import argparse
from matplotlib import pyplot
from tensorflow.keras.applications.inception_v3 import InceptionV3

from model import *
from train import *
from gan_utils import *
from data_utils import *
from gan_utils import *
from model_utils import *
from test_utils import prepare_real_images

# Parser creation
parser = argparse.ArgumentParser(
    description="Imposta i parametri per l'esecuzione del programma."
)
# Parser parameter definition
parser.add_argument("-z", "--redshift", type=float, required=True,
                    help="Redshift value (obbligatorio).")
parser.add_argument("--d_steps", type=int, default=5,
                    help="Discriminator steps / Generator steps (default: 5).")
parser.add_argument("-e", "--epochs", type=int, default=500,
                    help="Number of epochs (default: 500).")
parser.add_argument("--noise_dim", type=int, default=512,
                    help="Latent noise vector dimension (default: 512).")
parser.add_argument("-spe", "--steps_per_epoch", type=int, default=7,
                    help="Steps per epoch (default: 7).")
parser.add_argument("--end_size", type=int, default=128,
                    help="Target size of the images in the final step (default: 128).")
parser.add_argument("--batch_size", nargs="+", type=int, default=32,
                    help="Batch size (default: 32).")
# Arguments parsing
args = parser.parse_args()

def is_power_of_two(n):
    """Check if an input value n is a multiple of 2."""
    return n > 0 and (n & (n - 1)) == 0

# Validate the END_SIZE input value
if not is_power_of_two(args.end_size) or args.end_size > 512:
    print("Error: END_SIZE must be a power of two and less than 512")
    sys.exit(1)

# Print of parser parameters
print(f"REDSHIFT: {args.redshift}")
print(f"D_STEPS: {args.d_steps}")
print(f"EPOCHS: {args.epochs}")
print(f"NOISE_DIM: {args.noise_dim}")
print(f"STEPS_PER_EPOCH: {args.steps_per_epoch}")
print(f"END_SIZE: {args.end_size}")
print(f"BATCH_SIZE: {args.batch_size}")

REDSHIFT = args.redshift
D_STEPS = args.d_steps
EPOCHS = args.epochs
NOISE_DIM = args.noise_dim
STEPS_PER_EPOCH = args.steps_per_epoch
END_SIZE = args.end_size
BATCH_SIZE = args.batch_size

version = '_z_' + str(REDSHIFT)
CKPT_OUTPUT_PATH, IMG_OUTPUT_PATH, LOSS_OUTPUT_PATH = create_folders(version=version)

meta_data = load_meta_data(REDSHIFT, show=True)
print(f"Data Shape: {meta_data.shape}")

# Computing the Fid parameters associated with the real dataset
fid_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
mu1, sigma1 = prepare_real_images(fid_model=fid_model, meta_data=meta_data, target_size=END_SIZE)

pgan = PGAN(latent_dim = NOISE_DIM, d_steps = D_STEPS)
cbk = GANMonitor(num_img = len(meta_data), latent_dim = NOISE_DIM, redshift=REDSHIFT,
                 fid_model=fid_model, fid_real_par=(mu1, sigma1),
                 checkpoint_dir=CKPT_OUTPUT_PATH, image_path=IMG_OUTPUT_PATH)
cbk.set_steps(steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS) # 110, 6
cbk.set_prefix(prefix='0_init')

from tensorflow.python.client import device_lib
print("\n", device_lib.list_local_devices(), "\n")

# Local
pgan = train_fixed(0.001, 0.001, 0.001, EPOCHS, BATCH_SIZE, STEPS_PER_EPOCH, END_SIZE,
                   cbk, pgan, meta_data, loss_out_path=LOSS_OUTPUT_PATH)
# pgan = train(0.001, 0.001, 0.001, EPOCHS, BATCH_SIZE, STEPS_PER_EPOCH, END_SIZE,
                   # cbk, pgan, meta_data, loss_out_path=LOSS_OUTPUT_PATH)
# Save the values of the FID score at the end of training
np.save(f"fid_scores{version}", cbk.fid_scores)

# tstr = compute_tstr(meta_data= meta_data, model=pgan, d_steps=D_STEPS, NOISE_DIM=NOISE_DIM, END_SIZE=END_SIZE)
# print(f"TSTR: {tstr}")

