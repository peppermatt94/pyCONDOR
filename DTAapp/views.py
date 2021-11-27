from django.shortcuts import render
from django.template import loader
from django.conf import settings
from django.template import RequestContext
from django.http import HttpResponse
from bokeh.embed import components
from .models import Sample, InputSample
from .forms import DTAselection
from .plotting import myfig
from .main import main, interpolate_only
import pandas as pd
import json

def DTA_webAPP(request):
    context={}
    result = None
    imps = pd.DataFrame(columns=["1","2","3","4","5"]) # i need this to avoid problem in create_grid in plotting.py
    dta = pd.DataFrame(columns = ["3","4"])
    fig = myfig(imps,dta)
    script,div = components(fig.layout) 
    form = DTAselection(request.POST,request.FILES)
    if request.method == 'POST' and "Compute" in request.POST:
        if form.is_valid():
            data = pd.read_csv(request.FILES["ImportData"],sep = "\t",  skiprows = [1,2])
            imps, dta = main(data,float(form['treshold_derivative'].value()), form['method_of_discretization'].value(),float(form['regularization_parameter'].value()), int(form['regularization_order'].value()), form['type_of_interpolation'].value() , int(form['number_of_point'].value()), form['col_selection'].value())
            fig = myfig(imps, dta)
            script, div = components(fig.layout)
            imps.to_csv("result_imps.csv", index = False)
            dta.to_csv("result_dta.csv", index = False)
            result = "Ok, boy"
    if request.method == 'POST' and "Show interpolation" in request.POST:
        if form.is_valid():
            data = pd.read_csv(request.FILES["ImportData"],sep = "\t",  skiprows = [1,2])
            real, imag,omega_vec , dxdy= interpolate_only(data,float(form['treshold_derivative'].value()), form['type_of_interpolation'].value() , int(form['number_of_point'].value()), form['col_selection'].value())
            imps = pd.DataFrame({"freq":omega_vec, "H'":real, "H''":imag, "H1'":real, "H1''":imag})
            dta = pd.DataFrame({"freq":omega_vec, "real":real})
            fig = myfig(imps,dta)
            script, div = components(fig.layout)
    context =  {'form': form, 'script':script, 'div':div, "result":result}
    return HttpResponse(render(request, "dta_webapp.html", context))

def download_result(request):
    response = HttpResponse(open("result_dta.csv", 'rb').read())
    response['Content-Type'] = 'text/plain'
    response['Content-Disposition'] = 'attachment; filename=dta.csv'
    return response

def download_fit(request):
    response = HttpResponse(open("result_imps.csv", 'rb').read())
    response['Content-Type'] = 'text/plain'
    response['Content-Disposition'] = 'attachment; filename=imps.csv'
    return response

def documentation(request):
    return HttpResponse(render(request, "doc.html"))

def source_code(request):
    #code_paths = Path('FastProgram/.')
    commit_path = settings.CODE_PATHS / "commit.json"
    with open(commit_path.as_posix(), "r") as f:
        data = json.load(f)

    codes = [i for i in data["Files"]]
    context = {"codes" : codes}
    return HttpResponse(render(request, "source.html", context))

def functionpy(request, filename):
    file_path = list(settings.CODE_PATHS.glob(filename))[0].as_posix()
    contents = open(file_path).read()
    final_contents = "<pre>" + contents + "</pre>"
    with open("DTAapp/templates/code.html", "w") as e:
        e.write(final_contents)
    return HttpResponse(render(request, "code.html"))


