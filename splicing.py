from config import get_arguments
#from SinGAN.manipulate import *
#from SinGAN.training import *
import SinGAN.functions as functions
import SinGAN.splicing_functions as sf


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Splicing')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--original_name', help='defines the folder name', default='')

    
    parser.add_argument('--mode', help='task to be done', default='splicing_train')

    
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    opt.original_name = opt.input_name  #used to call the folder the correct name for the original image
    sf.readimage(opt)

    print(opt.input_name)
    #change input directory to point to spliced images
    opt.input_dir = "%s/%s" % (opt.input_dir, opt.original_name[:-4])
    for i in range(1, 5):
        opt.mode = 'splicing_train'
        opt.input_name = "%s_%d.png" % (opt.original_name[:-4], i)
        print("Iteration %d" % (i))
        print(opt.input_dir)
        print(opt.input_name)
        sf.spliceTrain(opt)

    #get random sample images and splice them together

    """
    images = []
    for i in range(1,5):        #iterate over the 4 slices
        images.append([])
        for j in range(50):     #iterate over the 50 random samples
            filepath = '%s/RandomSamples_Splicing/%s/%s%d/gen_start_scale=%d' % (opt.out, opt.original_name[:-4], opt.input_name[:-5], i, opt.gen_start_scale)
            images[i].append(filepath)
    print("image list")
    for slice in images:
        for x in range(3):
            print(slice[x])

    #'%s/RandomSamples_Splicing/%s/%s/gen_start_scale=%d' % (opt.out, opt.original_name[:-4], opt.input_name[:-4], opt.gen_start_scale)
    #opt.gen_start_scale
    #SinGAN\Output\RandomSamples_splicing\volcano_100px\volcano_100px_1\gen_start_scale=0
    """
    

