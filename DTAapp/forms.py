from django import forms
from django.conf import settings
discretization_methods =(
    ("1", "Gaussian"),
("2", "Inverse Quadric"),
("3", "Inverse Quadratic")
    )
orders = (
        ("1","First order"),
        ("2", "Second order")
        )

class DTAselection(forms.Form):
    ImportData = forms.FileField( label = "Import Data")
    method_of_discretization = forms.ChoiceField(choices = discretization_methods, label = "Select a discretization Method")
    regularization_parameter = forms.FloatField(label = "Regularization Parameter")#, widget=forms.TextInput(attrs={'type':'number'}), initial = 1)
    number_of_point = forms.IntegerField(label = "Number of interpolation Point",  widget=forms.TextInput(attrs={'type':'number'}))
    regularization_order = forms.ChoiceField(label = "Tichonov regularization order", choices = orders, initial = "2")
    col_selection = forms.CharField(label = "Choice the columns to take into account in file in python style (for instance 1:3 for 1,2 column)")

    
