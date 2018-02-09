%Image cropping with user interface
%Mona Omidyeganeh Jan. 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;clear all;
%inputs:
%the text file that contain name of the images
flist='/Users/mona/Documents/Mcgill_Brain/receptor_registration/cropping/img_list.txt';
%input folder:where the original files are
inputfolder='/Users/mona/Documents/Mcgill_Brain/receptor_registration/cropping/';
%output folder:where to save the cropped images
outputfolder='/Users/mona/Documents/Mcgill_Brain/receptor_registration/cropping/cropped_images/';
%border pixels of each region
lb=50; %increase it if you want to have more borders e.g. 50 or 100
er_size=5;%disk size for imerode
di_size=5;%disk size for dilation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fnames=textread(flist, '%s');
warning off
for i=1:length(fnames)
    i
    close all;
    im1=imread([num2str(inputfolder),num2str(fnames{i})]);
    [im7,a,b]=cropp_img(im1,er_size,di_size,lb);
%     figure,subplot(2,3,1),imshow(im1);
%     subplot(2,3,2),imshow(im2);subplot(2,3,3),imshow(uint8(im3.*255));    
%     subplot(2,3,3),imshow(uint8(im4.*255));
%     subplot(2,3,4),imshow(uint8(im5.*255));
%     subplot(2,3,5),imshow(uint8(im6.*255));
%     subplot(2,3,6),imshow(im7);
    %user interface:
    im8(:,:,1)=im1;im8(:,:,2)=im1;im8(:,:,3)=im1;
    %
    im8(max(1,min(a)-lb):min(max(a)+lb,size(im1,1)),max(1,min(b)-lb-10):max(1,min(b)-lb)+15,3)=0;
    im8(max(1,min(a)-lb):min(max(a)+lb,size(im1,1)),min(max(b)+lb,size(im1,2))-10:min(max(b)+lb,size(im1,2))+15,3)=0;
    im8(max(1,min(a)-lb-10):max(1,min(a)-lb)+15,max(1,min(b)-lb):min(max(b)+lb,size(im1,2)),3)=0;
    im8(min(max(a)+lb,size(im1,1))-10:min(max(a)+lb+5,size(im1,1)),max(1,min(b)-lb):min(max(b)+lb,size(im1,2)),3)=0;
    %
    im8(max(1,min(a)-lb):min(max(a)+lb,size(im1,1)),max(1,min(b)-lb-10):max(1,min(b)-lb)+15,2)=0;
    im8(max(1,min(a)-lb):min(max(a)+lb,size(im1,1)),min(max(b)+lb,size(im1,2))-10:min(max(b)+lb,size(im1,2))+15,2)=0;
    im8(max(1,min(a)-lb-10):max(1,min(a)-lb)+15,max(1,min(b)-lb):min(max(b)+lb,size(im1,2)),2)=0;
    im8(min(max(a)+lb,size(im1,1))-10:min(max(a)+lb,size(im1,1))+15,max(1,min(b)-lb):min(max(b)+lb,size(im1,2)),2)=0;
    %
    im8(max(1,min(a)-lb):min(max(a)+lb,size(im1,1)),max(1,min(b)-lb-10):max(1,min(b)-lb)+15,1)=255;
    im8(max(1,min(a)-lb):min(max(a)+lb,size(im1,1)),min(max(b)+lb,size(im1,2))-10:min(max(b)+lb,size(im1,2))+15,1)=255;
    im8(max(1,min(a)-lb-10):max(1,min(a)-lb)+15,max(1,min(b)-lb):min(max(b)+lb,size(im1,2)),1)=255;
    im8(min(max(a)+lb,size(im1,1))-10:min(max(a)+lb,size(im1,1))+15,max(1,min(b)-lb):min(max(b)+lb,size(im1,2)),1)=255;
    figure,imshow(im8);
    
    choice = questdlg('Is the cropping correct?','Yes','No');
    % Handle response
    switch choice
        case 'No'
            I=imshow(im1);
            im9=imcrop(im1);
            imshow(im9);
        case 'Yes'
            im9=im7;
        
    end
    close all
    %save response
    imwrite(im9,[num2str(outputfolder),num2str(fnames{i})]);
    clear im1 im2 im3 im4 im5 im6 im7 im8 im9
end

