function Iy = single_image_preprocess_finetuning(img_url)

I = imread(img_url);


Iy = imresize3(I,[224,224,3]);

end