from django.db import models
from django.forms import ModelForm

# Create your models here.

class Sample(models.Model):
    sample_id = models.CharField(max_length =200)
    pub_date = models.DateTimeField('data published')
    #Cigs_Width = models.CharField(max_length = 200)
    #Cds_Width = models.CharField(max_length = 200)
    characteristics = models.CharField(max_length=200)
    upload = models.FileField(upload_to= "uploads", default = None) 
    def __str__(self):
        return self.sample_id
    
class InputSample(ModelForm):
    class Meta:
        model = Sample
        fields = ['sample_id', 'pub_date', 'characteristics', 'upload' ]
