import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import imageio
from glob import glob

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 



ligands = ["ampa","kain","mk80","ly34","musc","sr95","cgp5","flum","pire","oxot","afdx","damp","epib","praz","uk14","rx82","dpat","keta","sch2", "dpmg"]
receptor = {"ampa":'AMPA',"kain":'Kainate',"mk80":'NMDA',"ly34":'mGluR2/3',"musc":'A (Agonist)',"sr95":'A (Antagonist)',"cgp5":'B',"flum":'A (Benzodiazepine)',"pire":'Musc. M1',"oxot":'Musc. M2 (Agonist)',"afdx":'Musc. M2 (Antagonist)',"damp":'Musc. M3',
"epib":'Nicotinic '+u'\u03B1'+"4"+u'\u03B2'+"2",
"praz":u'\u03B1' +'1',
"uk14":u'\u03B1' + '2 (Agonist)',
"rx82":u'\u03B1' + '2 (Antagonist)',
"dpat":'5-HT1A',
"keta":'5-HT2',
"sch2":'D1',
"racl":'D2', 
'dpmg':'Adenosine 1'}

transmitter = {"ampa":'Glutamate',"kain":'Glutamate',"mk80":'Glutamate',"ly34":'Glutamate',"musc":'GABA',"sr95":'GABA',"cgp5":'GABA',"flum":'GABA',"pire":'Acetylcholine',"oxot":'Acetylcholine',"afdx":'Acetylcholine',"damp":'Acetylcholine',"epib":'Acetylcholine' ,"praz":'Noradrenalin',"uk14":'Noradrenalin',"rx82":'Noradrenalin',"dpat":'Serotonin',"keta":'Serotonin',"sch2":'Dopamine','racl':'Dopamine',"dpmg":'Adenosine'}
d=6
for i, ligand in enumerate(ligands) : 
    f=glob("lin/R_slab_1/*"+ligand+"*15#L.TIF")[0]
    tt=str(transmitter[ligand] +' / '+receptor[ligand])
    print(i+1, tt)
    img = Image.open(f)
    img = img.resize((int(img.size[0]/d),int(img.size[1]/d)),Image.ANTIALIAS)
    ar= np.array(img.getdata())
    threshold = np.percentile( ar, [99] )[0]
    ar[ ar > threshold] = threshold
    ar = 256* (ar-np.min(ar))/(np.max(ar)-np.min(ar))
    img.putdata(ar)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 40)

    button_img = Image.new('I', (len(tt)*25,50), "black")
    button_draw = ImageDraw.Draw(button_img)
    button_draw.text((0, 0), tt, font=font,  fill="white")

    # put button on source image in position (0, 0)
    img.paste(button_img, (0, 0))
    img.save('example_'+ligand+'.jpg')


    #plt.title( tt, color='white', size=18)
    #plt.imshow(img, cmap=plt.cm.gray); 
    #plt.tight_layout()
    #plt.savefig('example_'+ligand+'.jpg', facecolor='black')

