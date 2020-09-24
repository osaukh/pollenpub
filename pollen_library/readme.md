# Dataset description
This dataset contains microscopic images and videos of pollen gathered between Feb. and Aug. 2020 in Graz, Austria.
- Pollen images of 16 types: `images_16_types.zip` 
	- Acer Pseudoplatanus
	- Aesculus Carnea
	- Alnus
	- Anthoxanthum
	- Betula Pendula
	- Brassica
	- Carpinus
	- Corylus
	- Dactylis Glomerata
	- Fraxinus
	- Pinus Nigra	
	- Platanus
	- Populus Nigra
	- Prunus Avium
	- Sequoiadendron Giganteum
	- Taxus Baccata
	
- Pollen video library `pollen_video_library.zip`
	- Each type of pollen is in a separate folder, there may be multiple videos per type.
	- In each pollen folder, we included images cropped from the videos by YOLO object detection algorithm trained on a subset of pollen images as described in [1].

- Field data over 3 days are gathered in Graz in spring 2020. `pollen_field_data.zip`

- Sample code to load the data and visualize the images is in `plot_pollen_sample.py`. Download and extract the file 1images_16_types.zip1 in the same folder as `plot_pollen_sample.py` to run the example.


# Dependecies:
- opencv
- numpy
- matplotlib

# Credit
[1] N. Cao, M. Meyer, L. Thiele, and O. Saukh. 2020. Automated Pollen Detection with an Affordable Technology. In Proceedings of the International Conference on Embedded Wireless Systems and Networks (EWSN). 108–119.
```
@inproceedings{namcao2020pollen,
  title = {Automated Pollen Detection with an Affordable Technology},
  author = {Nam Cao and Matthias Meyer and Lothar Thiele and Olga Saukh},
  booktitle = {Proceedings of the International Conference on Embedded Wireless Systems and Networks (EWSN)},
  pages={108–119}
  month = {2},	
  year = {2020},
}
```  
