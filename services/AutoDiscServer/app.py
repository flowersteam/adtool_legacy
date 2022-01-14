#### Add auto_disc lib to path ####
import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../../libs/"))
#### Add auto_disc lib to path ####

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from AutoDiscServer.experiments import ExperimentsHandler
from AutoDiscServer.utils import get_auto_disc_registered_modules_info, get_auto_disc_registered_callbacks, list_profiles
from AutoDiscServer.utils.DB.expe_db_utils import SavableOutputs

from auto_disc import REGISTRATION

app = Flask(__name__)
CORS(app)

# Experiments
experiments_handler = ExperimentsHandler() # Singleton handling experiments

@app.route('/experiments', methods=['GET'])
def list_experiments():
    experiment_ids = experiments_handler.list_running_experiments()
    return make_response(jsonify({
        "ID": experiment_ids
    }), 200)

@app.route('/experiments', methods=['POST'])
def create_experiment():
    try:
        id = experiments_handler.add_experiment(request.json)
        return make_response(jsonify({'ID': id}), 200)
    except Exception as ex:
        error_message = "Error while creating the new experiment : {}".format(ex)
        print(error_message)
        return make_response(error_message, 403)

@app.route('/experiments/<int:id>', methods=['DELETE'])
def stop_experiment(id):
    try:
        experiments_handler.remove_experiment(id)
        return make_response(jsonify({'ID': id}), 200)
    except Exception as ex:
        error_message = "Error while deleting experiment with id {}: {}".format(id, ex)
        print(error_message)
        return make_response(error_message, 403)

# Explorers
@app.route('/explorers', methods=['GET'])
def list_explorers():
    info = get_auto_disc_registered_modules_info(REGISTRATION['explorers'])
    return make_response(
        jsonify(info), 
    200)

# Systems
@app.route('/systems', methods=['GET'])
def list_systems():
    info = get_auto_disc_registered_modules_info(REGISTRATION['systems'])
    return make_response(
        jsonify(info), 
    200)

# Output Representations
@app.route('/output-representations', methods=['GET'])
def list_output_representations():
    info = get_auto_disc_registered_modules_info(REGISTRATION['output_representations'])
    return make_response(
        jsonify(info), 
    200)

# Input Wrappers
@app.route('/input-wrappers', methods=['GET'])
def list_input_wrappers():
    info = get_auto_disc_registered_modules_info(REGISTRATION['input_wrappers'])
    return make_response(
        jsonify(info), 
    200)

# On discovery callback
@app.route('/discovery-saving-keys', methods=['GET'])
def list_keys_to_save_on_discovery():
    info = [member.value for member in list(SavableOutputs)]
    return make_response(
        jsonify(info), 
    200)

# Callbacks
@app.route('/callbacks', methods=['GET'])
def list_callbacks():
    info = get_auto_disc_registered_callbacks(REGISTRATION['callbacks'])
    return make_response(
        jsonify(info), 
    200)

# Hosts on which experiments can be run
@app.route('/hosts', methods=['GET'])
def list_hosts():
    profiles = ['local']
    remote_profiles = [profile[0] for profile in list_profiles()]
    profiles.extend(remote_profiles)
    return make_response(
        jsonify(profiles), 
    200)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)