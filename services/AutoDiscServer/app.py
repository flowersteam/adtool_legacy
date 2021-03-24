from flask import Flask, request, jsonify, make_response

app = Flask(__name__)


# Experiments
# TODO: Create a singleton handling experiments
@app.route('/experiments', methods=['GET'])
def list_experiments():
	return make_response(jsonify({}), 200)

@app.route('/experiments', methods=['POST'])
def create_experiment():
    try:
        print(request.form)
        raise NotImplementedError()
        return make_response(jsonify({}), 200)
    except Exception as ex:
        error_message = "Error while creating the new experiment : {}".format(ex)
        print(error_message)
        return make_response(error_message, 403)

@app.route('/experiments/<int:id>', methods=['GET'])
def get_experiment(id):
    try:
        print(id)
        raise NotImplementedError()
        return make_response(jsonify({}), 200)
    except Exception as ex:
        error_message = "Error while fetching experiment with id {}: {}".format(id, ex)
        print(error_message)
        return make_response(error_message, 403)

# Explorers
@app.route('/explorers', methods=['GET'])
def list_explorers():
	return make_response(jsonify({}), 200)

# Systems
@app.route('/systems', methods=['GET'])
def list_systems():
	return make_response(jsonify({}), 200)

# Output Representations
@app.route('/output-representations', methods=['GET'])
def list_output_representations():
	return make_response(jsonify({}), 200)

# Input Wrappers
@app.route('/input-wrappers', methods=['GET'])
def list_input_wrappers():
	return make_response(jsonify({}), 200)

if __name__ == '__main__':
	app.run()