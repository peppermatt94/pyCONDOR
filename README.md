# pyCONDOR: Analyses of Intensity Modulated Photocurrent Spectroscopy

The presented tool aim at resolving the characteristic time of the processes behind an Intensity Modulated Photocurrent Spectroscopy (IMPS) spectrum. IMPS spectrum is transformed in a Distribution of Time Admittance (DTA) spectrum. This is peaked in values of time connected to characteristic times of the processes embedded in the charge carrier dynamic. 

 ![Alt Text](https://github.com/peppermatt94/pyCONDOR/blob/master/dta.gif)

## Project architecture

The program is developed with a **django** architecture: the directory *DTAtools* contains the settings for the web interface, while DTAapp contains the script needed for the execution of the program. 

Within this directory, the file *views.py* is the script that invokes the web response and call the html and the forms of page. Forms are created in *forms.py* and the html codes are in the *template* directory. After the filling of forms, `compute` button invoke the functions within the *main.py* script. Here the uploaded file is read with the classmethod `fromfile` of the class `IMPS` constructed in the *DTA_main.pyi*. The function `compute_bokeh` is constructed in the *compute.py* script. In this last file the computation of the program is done. The computation is done thanks to the core functions of the program present in *dtatools.py*. For a detailed explanation of the code I suggest my [master thesis](https://www.researchgate.net/publication/359344384_Growth_and_characterization_of_a_CuIn_07_Ga_03_Se_2_CdS_photocathode) .

## Forms

The forms for the user input are 6: input file, radial basis function type, regularization parameter, number of interpolation points, data in the file to be used. 

### input file

The input file must be a file with
 ![Alt Text](https://github.com/peppermatt94/pyCONDOR/blob/master/file.jpg)


  
