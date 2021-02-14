from django.shortcuts import render, redirect
from .predict import *

# Create your views here.

def home(request):
    if request.method == 'POST':
        fullname = request.POST.get('fullname')
        keywords = request.POST.get('keywords')
        search_string = fullname + ' , ' + keywords
        op_list = predict_traits(search_string)
        prediction, html = op_list[0], op_list[1] 
        return redirect('predictions')
    return render(request, 'predictor/index.html', {'predicted': False})

def prediction(request):
    return render(request, 'predictor/prediction.html')
