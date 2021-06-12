from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
import time


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='hr_train')
    parser.add_argument('--auto_lk', help='Automatic layer/kernel init', default=True)
    parser.add_argument('--size', help='Image size', default=400)
    
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    #set max size to size
    #set min size to size//10   opt.min_size = 
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        t0 = time.time()
        train(opt, Gs, Zs, reals, NoiseAmp)
        t1 = time.time()
        print("train_time:",round((t1-t0)/60,2), "min")
        SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)
