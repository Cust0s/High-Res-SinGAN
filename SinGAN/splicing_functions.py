from PIL import Image
import os
import sys
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions

def readimage(opt):
    savePath = '%s/%s/' % (opt.input_dir, opt.original_name[:-4])

    if (os.path.exists(savePath)):
        sys.exit("Image has already been spliced")
    else:
        try:
            os.makedirs(savePath)
        except OSError:
            pass
            
    img = Image.open('%s/%s' % (opt.input_dir,opt.input_name))

    width, height = img.size
    
    width = width // 2
    height = height // 2

    boundary = (0, 0, width, height)
    crop1 = img.crop(boundary)
    
    boundary = (width, 0, width * 2, height)
    crop2 = img.crop(boundary)
    
    boundary = (0, height, width, height * 2)
    crop3 = img.crop(boundary)
    
    boundary = (width, height, width * 2, height * 2)
    crop4 = img.crop(boundary)
    
    crop1.save(savePath + opt.original_name[:-4] + "_1.png", "PNG")
    crop2.save(savePath + opt.original_name[:-4] + "_2.png", "PNG")
    crop3.save(savePath + opt.original_name[:-4] + "_3.png", "PNG")
    crop4.save(savePath + opt.original_name[:-4] + "_4.png", "PNG")
    
    
def spliceTrain(opt):
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    print(opt.original_name)
    print(opt.input_name)
    print(dir2save)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        print("before training")
        print(dir2save)
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, NoiseAmp)
        SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    