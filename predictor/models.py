from django.db import models

# Create your models here.
class Prediction(models.Model):
    keywords = models.CharField(max_length=200, null=True)
    image = models.ImageField(upload_to='predictions')