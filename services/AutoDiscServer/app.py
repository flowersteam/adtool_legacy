from flask import Flask, request, jsonify, make_response
from AutoDiscServer.experiments_handler import ExperimentsHandler
from AutoDiscServer.utils import get_auto_disc_registered_modules_info
from auto_disc import REGISTRATION

app = Flask(__name__)

# Experiments
experiments_handler = ExperimentsHandler() # Singleton handling experiments

@app.route('/experiments', methods=['GET'])
def list_experiments():
    expeperiment_ids = experiments_handler.list_running_experiments()
    return make_response(jsonify({
        "ID": expeperiment_ids
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

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)