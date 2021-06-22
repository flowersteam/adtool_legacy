from flask import Flask, request, jsonify, make_response, send_file
from pymongo import MongoClient
from gridfs import GridFS
from bson.objectid import ObjectId

app = Flask(__name__)
client = MongoClient('mongodb://localhost:27017/', username="autodisc", password="password")
db = client.main_db
fs = GridFS(db)

def _get_one_by_filter(collection, filter):
    element = collection.find_one(filter)
    if element:
        element['_id'] = str(element['_id'])
        return make_response(jsonify(element), 200)
    else:
        return make_response("No element found matching {} in collection {}".format(filter, collection), 403)

#################################
########## DISCOVERIES ##########
# LIST
@app.route('/discoveries', methods=['GET'])
def list_discoveries():
    checkpoint_id = int(request.args.get('checkpoint_id', default=None))
    if checkpoint_id:
        discoveries = []
        for discovery in db.discoveries.find({"checkpoint_id": checkpoint_id}):
            discovery['_id'] = str(discovery['_id'])
            discoveries.append(discovery)
        return make_response(jsonify(discoveries), 200)
    else:
        return make_response("You must provide a checkpoint_id in the request args", 403)

# GET ONE
@app.route('/discoveries/<id>', methods=['GET'])
def get_discovery_by_id(id):
    return _get_one_by_filter(db.discoveries, {"_id": ObjectId(id)})

# ADD ONE
@app.route('/discoveries', methods=['POST'])
def create_discovery():
    discovery_to_insert = request.json
    added_discovery_id = db.discoveries.insert_one(discovery_to_insert).inserted_id
    return make_response(jsonify({"ID": str(added_discovery_id)}), 200)

# GET A FILE
@app.route('/discoveries/<id>/<file>', methods=['GET'])
def get_discovery_file(id, file):
    discovery = db.discoveries.find_one({"_id": ObjectId(id)})
    if discovery:
        if not file in discovery: 
            return make_response("No file {} found with in discovery with id {}".format(file, id), 403)
        file = fs.get(ObjectId(discovery[file]))
        return send_file(file, attachment_filename=file.filename)
    else:
        return make_response("No discovery found with id {}".format(id), 403)

# ADD FILES
@app.route('/discoveries/<id>/files', methods=['POST'])
def add_discovery_files(id):
    discovery = db.discoveries.find_one({"_id": ObjectId(id)})
    if discovery:
        # Add files in GridFS
        updates_to_do = {'$set': {}}
        for file_name in request.files:
            file_id = fs.put(request.files[file_name], filename=request.files[file_name].filename)
            updates_to_do['$set'][file_name] = str(file_id)
        # Add GridFS ids and filenames to discovery
        db.discoveries.update_one({"_id": discovery["_id"]}, updates_to_do)
        return make_response(jsonify({"OK"}), 200)
    else:
        return make_response("No discovery found with id {}".format(id), 403)

#################################
########## EXPLORERS ##########
# LIST
@app.route('/explorers', methods=['GET'])
def list_explorers():
    checkpoint_id = int(request.args.get('checkpoint_id', default=None))
    if checkpoint_id:
        explorers = []
        for explorer in db.explorers.find({"checkpoint_id": checkpoint_id}):
            explorer['_id'] = str(explorer['_id'])
            explorers.append(explorer)
        return make_response(jsonify(explorers), 200)
    else:
        return make_response("You must provide a checkpoint_id in the request args", 403)
# GET ONE
@app.route('/explorers/<id>', methods=['GET'])
def get_explorer_by_id(id):
    return _get_one_by_filter(db.explorers, {"_id": ObjectId(id)})

# ADD ONE
@app.route('/explorers', methods=['POST'])
def create_explorer():
    explorer_to_insert = request.json
    added_explorer_id = db.explorers.insert_one(explorer_to_insert).inserted_id
    return make_response(jsonify({"ID": str(added_explorer_id)}), 200)

# GET A FILE
@app.route('/explorers/<id>/<file>', methods=['GET'])
def get_explorer_file(id, file):
    explorer = db.explorers.find_one({"_id": ObjectId(id)})
    if explorer:
        if not file in explorer: 
            return make_response("No file {} found with in explorer with id {}".format(file, id), 403)
        file = fs.get(ObjectId(explorer[file]))
        return send_file(file, attachment_filename=file.filename)
    else:
        return make_response("No explorer found with id {}".format(id), 403)

# ADD FILES
@app.route('/explorers/<id>/files', methods=['POST'])
def add_explorer_files(id):
    explorer = db.explorers.find_one({"_id": ObjectId(id)})
    if explorer:
        # Add files in GridFS
        updates_to_do = {'$set': {}}
        for file_name in request.files:
            file_id = fs.put(request.files[file_name], filename=request.files[file_name].filename)
            updates_to_do['$set'][file_name] = str(file_id)
        # Add GridFS ids and filenames to explorer
        db.explorers.update_one({"_id": explorer["_id"]}, updates_to_do)
        return make_response(jsonify({"OK"}), 200)
    else:
        return make_response("No explorer found with id {}".format(id), 403)

#############################
########## SYSTEMS ##########
# LIST
@app.route('/systems', methods=['GET'])
def list_systems():
    checkpoint_id = int(request.args.get('checkpoint_id', default=None))
    if checkpoint_id:
        systems = []
        for system in db.systems.find({"checkpoint_id": checkpoint_id}):
            system['_id'] = str(system['_id'])
            systems.append(system)
        return make_response(jsonify(systems), 200)
        # return _get_one_by_filter(db.systems, {"checkpoint_id": checkpoint_id})
    else:
        return make_response("You must provide a checkpoint_id in the request args", 403)

# GET ONE
@app.route('/systems/<id>', methods=['GET'])
def get_system_by_id(id):
    return _get_one_by_filter(db.systems, {"_id": ObjectId(id)})

# ADD ONE
@app.route('/systems', methods=['POST'])
def create_system():
    system_to_insert = request.json
    added_system_id = db.systems.insert_one(system_to_insert).inserted_id
    return make_response(jsonify({"ID": str(added_system_id)}), 200)

# GET A FILE
@app.route('/systems/<id>/<file>', methods=['GET'])
def get_system_file(id, file):
    system = db.systems.find_one({"_id": ObjectId(id)})
    if system:
        if not file in system: 
            return make_response("No file {} found with in system with id {}".format(file, id), 403)
        file = fs.get(ObjectId(system[file]))
        return send_file(file, attachment_filename=file.filename)
    else:
        return make_response("No system found with id {}".format(id), 403)

# ADD FILES
@app.route('/systems/<id>/files', methods=['POST'])
def add_system_files(id):
    system = db.systems.find_one({"_id": ObjectId(id)})
    if system:
        # Add files in GridFS
        updates_to_do = {'$set': {}}
        for file_name in request.files:
            file_id = fs.put(request.files[file_name], filename=request.files[file_name].filename)
            updates_to_do['$set'][file_name] = str(file_id)
        # Add GridFS ids and filenames to system
        db.systems.update_one({"_id": system["_id"]}, updates_to_do)
        return make_response(jsonify({"OK"}), 200)
    else:
        return make_response("No system found with id {}".format(id), 403)

####################################
########## INPUT_WRAPPERS ##########
# LIST
@app.route('/input_wrappers', methods=['GET'])
def list_input_wrappers():
    checkpoint_id = int(request.args.get('checkpoint_id', default=None))
    if checkpoint_id:
        input_wrappers = []
        for input_wrapper in db.input_wrappers.find({"checkpoint_id": checkpoint_id}):
            input_wrapper['_id'] = str(input_wrapper['_id'])
            input_wrappers.append(input_wrapper)
        return make_response(jsonify(input_wrappers), 200)
    else:
        return make_response("You must provide a checkpoint_id in the request args", 403)

# GET ONE
@app.route('/input_wrappers/<id>', methods=['GET'])
def get_input_wrapper_by_id(id):
    return _get_one_by_filter(db.input_wrappers, {"_id": ObjectId(id)})

# ADD ONE
@app.route('/input_wrappers', methods=['POST'])
def create_input_wrapper():
    input_wrapper_to_insert = request.json
    added_input_wrapper_id = db.input_wrappers.insert_one(input_wrapper_to_insert).inserted_id
    return make_response(jsonify({"ID": str(added_input_wrapper_id)}), 200)

# GET A FILE
@app.route('/input_wrappers/<id>/<file>', methods=['GET'])
def get_input_wrapper_file(id, file):
    input_wrapper = db.input_wrappers.find_one({"_id": ObjectId(id)})
    if input_wrapper:
        if not file in input_wrapper: 
            return make_response("No file {} found with in input_wrapper with id {}".format(file, id), 403)
        file = fs.get(ObjectId(input_wrapper[file]))
        return send_file(file, attachment_filename=file.filename)
    else:
        return make_response("No input_wrapper found with id {}".format(id), 403)

# ADD FILES
@app.route('/input_wrappers/<id>/files', methods=['POST'])
def add_input_wrapper_files(id):
    input_wrapper = db.input_wrappers.find_one({"_id": ObjectId(id)})
    if input_wrapper:
        # Add files in GridFS
        updates_to_do = {'$set': {}}
        for file_name in request.files:
            file_id = fs.put(request.files[file_name], filename=request.files[file_name].filename)
            updates_to_do['$set'][file_name] = str(file_id)
        # Add GridFS ids and filenames to input_wrapper
        db.input_wrappers.update_one({"_id": input_wrapper["_id"]}, updates_to_do)
        return make_response(jsonify({"OK"}), 200)
    else:
        return make_response("No input_wrapper found with id {}".format(id), 403)

############################################
########## OUTPUT_REPRESENTATIONS ##########
# LIST
@app.route('/output_representations', methods=['GET'])
def list_output_representations():
    checkpoint_id = int(request.args.get('checkpoint_id', default=None))
    if checkpoint_id:
        output_representations = []
        for output_representation in db.output_representations.find({"checkpoint_id": checkpoint_id}):
            output_representation['_id'] = str(output_representation['_id'])
            output_representations.append(output_representation)
        return make_response(jsonify(output_representations), 200)
    else:
        return make_response("You must provide a checkpoint_id in the request args", 403)

# GET ONE
@app.route('/output_representations/<id>', methods=['GET'])
def get_output_representation_by_id(id):
    return _get_one_by_filter(db.output_representations, {"_id": ObjectId(id)})

# ADD ONE
@app.route('/output_representations', methods=['POST'])
def create_output_representation():
    output_representation_to_insert = request.json
    added_output_representation_id = db.output_representations.insert_one(output_representation_to_insert).inserted_id
    return make_response(jsonify({"ID": str(added_output_representation_id)}), 200)

# GET A FILE
@app.route('/output_representations/<id>/<file>', methods=['GET'])
def get_output_representation_file(id, file):
    output_representation = db.output_representations.find_one({"_id": ObjectId(id)})
    if output_representation:
        if not file in output_representation: 
            return make_response("No file {} found with in output_representation with id {}".format(file, id), 403)
        file = fs.get(ObjectId(output_representation[file]))
        return send_file(file, attachment_filename=file.filename)
    else:
        return make_response("No output_representation found with id {}".format(id), 403)

# ADD FILES
@app.route('/output_representations/<id>/files', methods=['POST'])
def add_output_representation_files(id):
    output_representation = db.output_representations.find_one({"_id": ObjectId(id)})
    if output_representation:
        # Add files in GridFS
        updates_to_do = {'$set': {}}
        for file_name in request.files:
            file_id = fs.put(request.files[file_name], filename=request.files[file_name].filename)
            updates_to_do['$set'][file_name] = str(file_id)
        # Add GridFS ids and filenames to output_representation
        db.output_representations.update_one({"_id": output_representation["_id"]}, updates_to_do)
        return make_response(jsonify({"OK"}), 200)
    else:
        return make_response("No output_representation found with id {}".format(id), 403)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5001)