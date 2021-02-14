from django.shortcuts import render, redirect
from .predict import *
from .models import Prediction

# Create your views here.

def home(request):
    if request.method == 'POST':
        fullname = request.POST.get('fullname')
        keywords = request.POST.get('keywords')
        search_string = fullname + ' , ' + keywords
        prediction = predict_traits(search_string)
        p = Prediction(image='predictor/prediction.png', keywords=keywords)
        p.save()
        ids = p.id
        p = Prediction.objects.get(id=ids)
        return render(request, 'predictor/prediction.html',{'path':p.image.path})
    return render(request, 'predictor/index.html')

def prediction(request):
    return render(request, 'predictor/prediction.html')
