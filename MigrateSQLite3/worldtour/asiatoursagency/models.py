from django.db import models

# Create your models here.
class Tour(models.Model):
    
    origin_country = models.CharField(max_length=64)