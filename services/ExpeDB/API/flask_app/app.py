from flask import Flask, request, jsonify, make_response, send_file
from flask_cors import CORS
from pymongo import MongoClient
from gridfs import GridFS
from bson.objectid import ObjectId
import json
from utils import ExpeDBConfig


config = ExpeDBConfig()
app = Flask(__name__)
CORS(app)
client = MongoClient('mongodb://{}:{}/'.format(config.MONGODB_HOST, config.MONGODB_PORT), username=config.MONGODB_USERNAME, password=config.MONGODB_PASSWORD)
db = client.main_db
fs = GridFS(db)

# Return request with a document matching filter 
def _get_one_by_filter(collection, filter, query=None):
    element = collection.find_one(filter, query)
    if element:
        element['_id'] = str(element['_id'])
        return make_response(jsonify(element), 200)
    else:
        return make_response("No element found matching {} in collection {}".format(filter, collection), 403)

# Return request with multiple documents matching filter 
def _get_multiple_by_filter(collection, filter, query=None):
    elements = []
    for element in collection.find(filter, query):
        element['_id'] = str(element['_id'])
        elements.append(element)
    return make_response(jsonify(elements), 200)

# Return request with a file from a document 
def _get_file_from_document(document, filename):
    if not filename in document: 
        return make_response("No file {} found with in document with id {}".format(filename, document['_id']), 403)
    file = fs.get(ObjectId(document[filename]))
    return send_file(file, attachment_filename=file.filename)


# Add filtes to a document
def _add_files_to_document(collection, document, files):
    # Add files in GridFS
    updates_to_do = {'$set': {}}
    for file_name in files:
        file_id = fs.put(files[file_name], filename=files[file_name].filename)
        updates_to_do['$set'][file_name] = str(file_id)
    # Add GridFS ids and filenames to document
    collection.update_one({"_id": document["_id"]}, updates_to_do)
    return make_response(jsonify({'success':True}), 200)


#################################
########## DISCOVERIES ##########
# GET
@app.route('/discoveries', methods=['GET']) # list discoveries given filter
def list_discoveries():
    filter = request.args.get('filter', default=None)
    if filter is not None:
        query = request.args.get('query', default=None)
        return _get_multiple_by_filter(db.discoveries, json.loads(filter), json.loads(query) if query else None)
    else:
        return make_response("You must provide a filter in the request args", 403)

@app.route('/discoveries/<id>', methods=['GET']) # get a discovery by its id
def get_discovery_by_id(id):
    return _get_one_by_filter(db.discoveries, {"_id": ObjectId(id)})

@app.route('/discoveries/<id>/<file>', methods=['GET']) # get file from a discovery by its name
def get_discovery_file(id, file):
    discovery = db.discoveries.find_one({"_id": ObjectId(id)})
    if discovery:
        return _get_file_from_document(discovery, file)
    else:
        return make_response("No discovery found with id {}".format(id), 403)

# POST
@app.route('/discoveries', methods=['POST']) # add a discovery
def create_discovery():
    added_discovery_id = db.discoveries.insert_one(request.json).inserted_id
    return make_response(jsonify({"ID": str(added_discovery_id)}), 200)

@app.route('/discoveries/<id>/files', methods=['POST']) # add files to a discovery
def add_discovery_files(id):
    discovery = db.discoveries.find_one({"_id": ObjectId(id)})
    if discovery:
        return _add_files_to_document(db.discoveries, discovery, request.files)
    else:
        return make_response("No discovery found with id {}".format(id), 403)

# DELETE
@app.route('/discoveries/<id>', methods=['DELETE']) # remove a discovery by its id
def delete_discovery_by_id(id):
    db.discoveries.delete_one({"_id": ObjectId(id)})
    return make_response(jsonify({'success':True}), 200)

@app.route('/discoveries/', methods=['DELETE']) # remove multiple discoveries given a checkpoint id
def delete_discoveries():
    checkpoint_id = int(request.args.get('checkpoint_id', default=None))
    if checkpoint_id is not None:
        db.discoveries.delete_many({"checkpoint_id": checkpoint_id})
        return make_response(jsonify({'success':True}), 200)
    else:
        return make_response("You must provide a checkpoint_id in the request args", 403)


######################################
########## CHECKPOINT SAVES ##########
# GET
@app.route('/checkpoint_saves', methods=['GET']) # list checkpoint saves given filter
def list_checkpoint_saves():
    filter = request.args.get('filter', default=None)
    if filter is not None:
        query = request.args.get('query', default=None)
        return _get_multiple_by_filter(db.checkpoint_saves, json.loads(filter), json.loads(query) if query else None)
    else:
        return make_response("You must provide a filter in the request args", 403)

@app.route('/checkpoint_saves/<id>', methods=['GET']) # get a checkpoint save by its id
def get_checkpoint_save_by_id(id):
    return _get_one_by_filter(db.checkpoint_saves, {"_id": ObjectId(id)})

@app.route('/checkpoint_saves/<id>/<file>', methods=['GET']) # get file from a checkpoint save by its name
def get_checkpoint_save_file(id, file):
    checkpoint_save = db.checkpoint_saves.find_one({"_id": ObjectId(id)})
    if checkpoint_save:
        return _get_file_from_document(checkpoint_save, file)
    else:
        return make_response("No checkpoint_save found with id {}".format(id), 403)

# POST
@app.route('/checkpoint_saves', methods=['POST']) # add a checkpoint save
def create_checkpoint_save():
    added_checkpoint_save_id = db.checkpoint_saves.insert_one(request.json).inserted_id
    return make_response(jsonify({"ID": str(added_checkpoint_save_id)}), 200)

@app.route('/checkpoint_saves/<id>/files', methods=['POST']) # add files to a checkpoint save
def add_checkpoint_save_files(id):
    checkpoint_save = db.checkpoint_saves.find_one({"_id": ObjectId(id)})
    if checkpoint_save:
        return _add_files_to_document(db.checkpoint_saves, checkpoint_save, request.files)
    else:
        return make_response("No checkpoint_save found with id {}".format(id), 403)

# DELETE
@app.route('/checkpoint_saves/<id>', methods=['DELETE']) # remove a checkpoint save by its id
def delete_checkpoint_save_by_id(id):
    db.checkpoint_saves.delete_one({"_id": ObjectId(id)})
    return make_response(jsonify({'success':True}), 200)

@app.route('/checkpoint_saves/', methods=['DELETE']) # remove multiple checkpoint save given a checkpoint id
def delete_checkpoint_saves():
    checkpoint_id = int(request.args.get('checkpoint_id', default=None))
    if checkpoint_id is not None:
        db.checkpoint_saves.delete_many({"checkpoint_id": checkpoint_id})
        return make_response(jsonify({'success':True}), 200)
    else:
        return make_response("You must provide a checkpoint_id in the request args", 403)

######################################
########## DATA SAVES ##########
# GET
@app.route('/data_saves', methods=['GET']) # list data saves given filter
def list_data_saves():
    filter = request.args.get('filter', default=None)
    if filter is not None:
        query = request.args.get('query', default=None)
        return _get_multiple_by_filter(db.data_saves, json.loads(filter), json.loads(query) if query else None)
    else:
        return make_response("You must provide a filter in the request args", 403)

@app.route('/data_saves/<id>', methods=['GET']) # get a data save by its id
def get_data_save_by_id(id):
    return _get_one_by_filter(db.data_saves, {"_id": ObjectId(id)})

@app.route('/data_saves/<id>/<file>', methods=['GET']) # get file from a data save by its name
def get_data_save_file(id, file):
    data_save = db.data_saves.find_one({"_id": ObjectId(id)})
    if data_save:
        return _get_file_from_document(data_save, file)
    else:
        return make_response("No data_save found with id {}".format(id), 403)

# POST
@app.route('/data_saves', methods=['POST']) # add a data save
def create_data_save():
    added_data_save_id = db.data_saves.insert_one(request.json).inserted_id
    return make_response(jsonify({"ID": str(added_data_save_id)}), 200)

@app.route('/data_saves/<id>/files', methods=['POST']) # add files to a data save
def add_data_save_files(id):
    data_save = db.data_saves.find_one({"_id": ObjectId(id)})
    if data_save:
        return _add_files_to_document(db.data_saves, data_save, request.files)
    else:
        return make_response("No data_save found with id {}".format(id), 403)
# PATCH
@app.route('/data_saves/<id>', methods=['PATCH']) # PATCH data
def patch_data_by_id(id):
    update_data = db.data_saves.find_one_and_update({"_id" : ObjectId(id)}, {"$set": request.get_json()})
    return make_response(jsonify({"data updated": str(update_data)}), 200)

# DELETE
@app.route('/data_saves/<id>', methods=['DELETE']) # remove a data save by its id
def delete_data_save_by_id(id):
    db.data_saves.delete_one({"_id": ObjectId(id)})
    return make_response(jsonify({'success':True}), 200)

@app.route('/data_saves/', methods=['DELETE']) # remove multiple data save given a data id
def delete_data_saves():
    data_id = int(request.args.get('data_id', default=None))
    if data_id is not None:
        db.data_saves.delete_many({"data_id": data_id})
        return make_response(jsonify({'success':True}), 200)
    else:
        return make_response("You must provide a data_id in the request args", 403)

if __name__ == '__main__':
	app.run(host=config.FLASK_HOST, port=config.FLASK_PORT)