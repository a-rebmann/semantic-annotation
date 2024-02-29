import pm4py
from pm4py.algo.discovery.ocel.ocpn import algorithm as ocpn_discovery
from pm4py.visualization.ocel.ocpn import visualizer as ocpn_visualizer
from pm4py.objects.log.log import EventLog
import os


def discover_oc_dfg(ocel_path, ocel_name):
    ocel = pm4py.read_ocel(os.path.join(ocel_path, ocel_name))
    ocdfg = pm4py.discover_ocdfg(ocel)
    pm4py.save_vis_ocdfg(ocdfg, os.path.join(ocel_path, ocel_name+".png"))
    # views the .model with the frequency annotation
    pm4py.view_ocdfg(ocdfg, format="png")


def discover_oc_pn(ocel_path, ocel_name):
    ocel = pm4py.read_ocel(os.path.join(ocel_path, ocel_name))
    model = ocpn_discovery.apply(ocel)
    # views the model with the frequency annotation
    gviz = ocpn_visualizer.apply(model, parameters={"format": "png"})
    ocpn_visualizer.view(gviz)