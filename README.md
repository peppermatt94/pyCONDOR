# pyCONDOR: Analyses of Intensity Modulated Photocurrent Spectroscopy

The presented tool aim at resolving the characteristic time of the processes behind an Intensity Modulated Photocurrent Spectroscopy (IMPS) spectrum. IMPS spectrum is transformed in a Distribution of Time Admittance (DTA) spectrum. This is peaked in values of time connected to characteristic times of the processes embedded in the charge carrier dynamic. 

 ![Alt Text](https://github.com/peppermatt94/pyCONDOR/blob/master/dta.gif)

## Project architecture

The program is developed with a **django** architecture: the directory *DTAtools* contains the settings for the web interface, while DTAapp contains the script needed for the execution of the program. 

Within this directory, the file *views.py* is the script that invokes the web response and call the html and the forms of page. Forms are created in *forms.py* and the html codes are in the *template* directory. After the filling of forms, `compute` button invoke the functions within the *main.py* script. Here the uploaded file is read with the classmethod `fromfile` of the class `IMPS` constructed in the *DTA_main.pyi*. The function `compute_bokeh` is constructed in the *compute.py* script. In this last file the computation of the program is done. The computation is done thanks to the core functions of the program present in *dtatools.py*. For a detailed explanation of the code I suggest my [master thesis](https://www.researchgate.net/publication/359344384_Growth_and_characterization_of_a_CuIn_07_Ga_03_Se_2_CdS_photocathode) .

## Forms

The forms for the user input are 6: input file, radial basis function type, regularization parameter, number of interpolation points, data in the file to be used. 

### input file

The input file must be a file as the following example:
 ![Alt Text](https://github.com/peppermatt94/pyCONDOR/blob/master/file.jpg)

It can contain also a lot of column (more IMPS) but there are the followin rule: 
 - The first column must be frequency column
 - The following column are alternating of Y' (real part) Y''(immaginary part) of the IMPS.

The rows of the files must be as follows:
- the first row is an header of names
- second row must be present and can contain whatever you want
- third row is the information you want to appear in a plot legend

From the third row on, there are the data. Try to be carefull to avoid missing values.

### Radial Basis Function

The radial basis function to use, I recommend to use *gaussian*.

### Regularization Parameter

When the Tikhonov regularization is performed, the weight to give to the regularization is the regularization parameter. 0.001 is a recommended value. 

### Number of interpolation points

The data are interpolated due to Discipline Convex Programming problem if this is not done. The number of interpolated points is suggested as 40. 

### Column to use 

I a file like that show above, there are a lot of data. You can select the dataset to use. The number you put is the number of the dataset in the sequence of the column, not the number of the column. Actually a dataset is identified by two columns. For instance, if in the file are presents three datasets (i.e. 7 columns: 1 column of frequency, real part of the first dataset, imaginary part of the second dataset and so on), if select 1 you select the first dataset. With 1:3 you select the first and second dataset, and so on. 
