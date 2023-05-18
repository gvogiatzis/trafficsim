import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sumo2real import RouteEditor

import typer
app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, add_completion=False)

@app.command("startgui")
def main(real_img_fname, sumo_routes_fname="sumo_routes.pk", save_file_name="real_routes.pk", load_file_name="real_routes.pk",
        files_path=os.path.join("output","routes")):
    if not os.path.exists(files_path):
        os.makedirs(files_path)

    sumo_routes_fname = os.path.join(files_path,sumo_routes_fname)
    save_file_name =    os.path.join(files_path,save_file_name)
    load_file_name =    os.path.join(files_path,load_file_name)
    if not os.path.exists(load_file_name):
        load_file_name = None

    route_editor = RouteEditor(real_img_fname=real_img_fname, sumo_routes_fname=sumo_routes_fname, save_file_name=save_file_name, load_file_name=load_file_name)
    route_editor.start_gui()

if __name__ == "__main__":
    app()

   


