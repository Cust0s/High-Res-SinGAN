from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
import SinGAN.splicing_functions as sf
import os
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
from PIL import Image


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Splicing')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--original_name', help='defines the folder name', default='')
    parser.add_argument('--original_dir', help='defines the folder name', default='')
    parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
    parser.add_argument('--editing_start_scale', help='editing injection scale', type=int, default=8)
    parser.add_argument('--ref_dir', help='input reference dir', default='')
    

    parser.add_argument('--mode', help='task to be done', default='splicing_train')

    opt = parser.parse_args()
    opt = functions.post_config(opt)
    opt.original_name = opt.input_name  #used to call the folder the correct name for the original image
    
    sf.readimage(opt)

    print(opt.input_name)
    #change input directory to point to spliced images
    opt.original_dir = opt.input_dir
    opt.input_dir = "%s/%s" % (opt.input_dir, opt.original_name[:-4])
    for i in range(1, 5):
        opt.mode = 'splicing_train'
        opt.input_name = "%s_%d.png" % (opt.original_name[:-4], i)
        print("Iteration %d" % (i))
        print(opt.input_dir)
        print(opt.input_name)
        sf.spliceTrain(opt)

    #train editing network
    opt.min_size = 50
    opt.max_size = 500
    opt.input_dir = "%s" % (opt.original_dir)
    opt.input_name = opt.original_name
    opt.mode = 'splicing_train'
    
    sf.spliceTrain(opt)
    
    #generate raw edit images
    #get random sample images and splice them together
    # generate lists of random gen slices
    images = []
    for i in range(1,5):        #iterate over the 4 slices
        images.append([])
        for j in range(50):     #iterate over the 50 random samples
            filepath = '%s/RandomSamples_Splicing/%s/%s_%d/gen_start_scale=%d/%d.png' % (opt.out, opt.original_name[:-4], opt.input_name[:-4], i, opt.gen_start_scale, j)
            images[i-1].append(filepath)
    
    
    print("image list")
    for slice in images:
        for x in range(5):
            print(slice[x])
  
    selected = [images[0][0],images[1][0],images[2][0],images[3][0]]
    for x in selected:
        print(x)
        
    # Todo: add automated mask creation

    # Todo: create a list of 4 selected images
    sf.genRawEdit(opt, selected)
    
    # ToDo: add option to create multiple images
    
    #Use trained model and editing image to edit image
    #get trained model here
    print("INFERENCE")
    opt.mode = 'splicing_editing'
    opt.ref_name = "%s_edit.png" % (opt.original_name[:-4])
    opt.input_name = "%s.png" % (opt.original_name[:-4])
    opt.ref_dir = opt.input_dir
    
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    #elif (os.path.exists(dir2save)):
    #    print("output already exist")
    else:
        try:
            os.makedirs(dir2save)           #generate output directory
        except OSError:
            pass
            
        real = functions.read_image(opt)
        real = functions.adjust_scales2image(real, opt)
        
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        if (opt.editing_start_scale < 1) | (opt.editing_start_scale > (len(Gs)-1)):
            print("injection scale should be between 1 and %d" % (len(Gs)-1))
        else:
            ref = functions.read_image_dir('%s/%s' % (opt.ref_dir, opt.ref_name), opt)
            mask = functions.read_image_dir('%s/%s_mask%s' % (opt.ref_dir,opt.ref_name[:-4],opt.ref_name[-4:]), opt)
            if ref.shape[3] != real.shape[3]:
                '''
                mask = imresize(mask, real.shape[3]/ref.shape[3], opt)
                mask = mask[:, :, :real.shape[2], :real.shape[3]]
                ref = imresize(ref, real.shape[3] / ref.shape[3], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]
                '''
                mask = imresize_to_shape(mask, [real.shape[2],real.shape[3]], opt)
                mask = mask[:, :, :real.shape[2], :real.shape[3]]
                ref = imresize_to_shape(ref, [real.shape[2],real.shape[3]], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]

            mask = functions.dilate_mask(mask, opt)
            
            N = len(reals) - 1
            n = opt.editing_start_scale
            in_s = imresize(ref, pow(opt.scale_factor, (N - n + 1)), opt)
            in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            in_s = imresize(in_s, 1 / opt.scale_factor, opt)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            plt.imsave('%s/start_scale=%d.png' % (dir2save, opt.editing_start_scale), functions.convert_image_np(out.detach()), vmin=0, vmax=1)
            print('%s/start_scale=%d.png' % (dir2save, opt.editing_start_scale))
            out = (1-mask)*real+mask*out
            plt.imsave('%s/start_scale=%d_masked.png' % (dir2save, opt.editing_start_scale), functions.convert_image_np(out.detach()), vmin=0, vmax=1)
            print('%s/start_scale=%d_masked.png' % (dir2save, opt.editing_start_scale))
