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

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, NoiseAmp)
        SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt, gen_start_scale=opt.gen_start_scale)
    
    
    
def genRawEdit(opt, imgs):
    print("genRawEdit")
    images = []
    for img in imgs:
        
        print(img)
        images.append(Image.open(img))

    #images = [Image.open(x) for x in imgs]
    
    width, height = images[0].size
    
    newWidth = width * 2
    newHeight = height * 2
    
    rawEditImg = Image.new('RGB', (newWidth, newHeight))
    rawEditImg.paste(images[0], (0,0))
    rawEditImg.paste(images[1], (width,0))
    rawEditImg.paste(images[2], (0,height))
    rawEditImg.paste(images[3], (width, height))

    rawEditImg.save('%s/%s_edit.png' % (opt.input_dir, opt.input_name[:-4]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    