from drpbx import *
# from drpbx import upload, move_file
import requests
import importlib
import json
import time
import os
import shutil
import traceback
import regex as re


def new_files_exist():
    new_files = False
    pulled_dir = "/home/ubuntu/Spooler_files_pulled/spooler_files"
    dst_dir = "/home/ubuntu/Spooler_files"
    if not os.path.isdir(pulled_dir):
        return new_files
    new_files = True
    pulled_files = [os.path.join(pulled_dir, fn) for fn in next(os.walk(pulled_dir))[2]]
    dst_files = [os.path.join(dst_dir, fn) for fn in next(os.walk(pulled_dir))[2]]
    for i in range(len(pulled_files)):
        src_path = pulled_files[i]
        dst_path = dst_files[i]
        shutil.copy(src_path, dst_path)
    shutil.rmtree(pulled_dir)
    return new_files

def update_in_database(result_dict, status_msg_dict, job_id):
    extracted_username = job_id.split("-")[2]
    requested_backend = job_id.split("-")[1]

    status_json_dir = (
        "/Backend_files/Status/" + requested_backend + "/" + extracted_username + "/"
    )
    status_json_name = "status-" + job_id + ".json"
    status_json_path = status_json_dir + status_json_name

    job_json_name = "job-" + job_id + ".json"
    job_json_start_path = "/Backend_files/Running_Jobs/" + job_json_name

    if status_msg_dict["status"] == "DONE":
        result_json_dir = (
            "/Backend_files/Result/"
            + requested_backend
            + "/"
            + extracted_username
            + "/"
        )
        result_json_name = "result-" + job_id + ".json"
        result_json_path = result_json_dir + result_json_name
        result_binary = json.dumps(result_dict).encode("utf-8")
        upload(DUMPSTRING=result_binary, DROPBOXPATH=result_json_path)
        finished_json_dir = (
            "/Backend_files/Finished_Jobs/"
            + requested_backend
            + "/"
            + extracted_username
            + "/"
        )
        job_json_final_path = finished_json_dir + job_json_name
        move_file(STARTPATH=job_json_start_path, FINALPATH=job_json_final_path)
    else:
        deleted_json_dir = "/Backend_files/Deleted_Jobs/"
        job_json_final_path = deleted_json_dir + job_json_name
        move_file(STARTPATH=job_json_start_path, FINALPATH=job_json_final_path)
    
    status_binary = json.dumps(status_msg_dict).encode("utf-8")
    upload(DUMPSTRING=status_binary, DROPBOXPATH=status_json_path)
    return

# parameters for server
username = "spooler"  #'synqs_test'#
password = "This_APP==*cool*"
server_domain = "http://coquma-sim.herokuapp.com/api/"  # "http://qsim-drop.herokuapp.com/"#
backends_list = ["fermions", "singlequdit", "multiqudit"]

# loop
while True:
    time.sleep(1)
    new_files = new_files_exist()
    if new_files:
        raise ValueError("New files must have come. So break to restart the program!")

    requested_backend = backends_list[0]
    backends_list.append(backends_list.pop(0))

    query_url = server_domain + requested_backend + "/get_next_job_in_queue/"
    queue_response = requests.get(
        query_url, params={"username": username, "password": password}
    )

    job_json_path = (queue_response.json())["job_json"]
    job_id = (queue_response.json())["job_id"]
    if job_json_path == "None":
        continue

    query_url = server_domain + requested_backend + "/get_job_status/"
    status_payload = {"job_id": job_id}
    queue_response = requests.get(
        query_url,
        params={
            "json": json.dumps(status_payload),
            "username": username,
            "password": password,
        },
    )
    status_msg_dict = queue_response.json()
    # print(job_json_path)

    job_json_dict = json.loads(get_file_content(DROPBOXPATH=job_json_path))

    requested_spooler = importlib.import_module(
        "spooler_" + requested_backend
    )  # __import__('spooler_' + requested_backend)
    add_job = getattr(requested_spooler, "add_job")
    result_dict = {}
    try:
        result_dict, status_msg_dict = add_job(job_json_dict, status_msg_dict)
    except Exception as ex:
        # Remove sensitive info like filepaths
        tb_list = traceback.format_exc().splitlines()
        for i in range(len(tb_list)):
            tb_list[i] = re.sub(r'File ".*[\\/]([^\\/]+.py)"', r'File "\1"', tb_list[i])
        slimmed_tb = ' '.join(tb_list)
        # Update status dict
        status_msg_dict["status"] = "ERROR"
        status_msg_dict["detail"] += "; " + slimmed_tb
        status_msg_dict["error_message"] += "; " + slimmed_tb
    update_in_database(result_dict, status_msg_dict, job_id)
