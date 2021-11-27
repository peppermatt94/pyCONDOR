from django import forms
from django.conf import settings
discretization_methods =(
    ("1", "Gaussian"),
("2", "C2 Matern"),
("3", "Inverse Quadratic")
    )
orders = (
        ("1","First order"),
        ("2", "Second order")
        )
interpolation = (
        ("1", "Simple interpolation"),
        ("2", "Double Region interpolation"),
        ("3", "Triple Region interpolation"),
        ("4", "No interpolation")
        )

class DTAselection(forms.Form):
    ImportData = forms.FileField( label = "Import Data")
    method_of_discretization = forms.ChoiceField(choices = discretization_methods, label = "Select a discretization Method")
    regularization_parameter = forms.FloatField(label = "Regularization Parameter")#, widget=forms.TextInput(attrs={'type':'number'}), initial = 1)
    number_of_point = forms.IntegerField(label = "Number of interpolation Point",  widget=forms.TextInput(attrs={'type':'number'}))
    treshold_derivative = forms.FloatField(label = "Treshold value of derivative at high frequency", initial = 0.005)
    type_of_interpolation = forms.ChoiceField(choices = interpolation, label = "Select a the type of interpolation of point", initial = "2")
    regularization_order = forms.ChoiceField(label = "Tichonov regularization order", choices = orders, initial = "2")
    col_selection = forms.CharField(label = "Choice the columns to take into account in file in python style (for instance 1:3 for 1,2 column)")
        
