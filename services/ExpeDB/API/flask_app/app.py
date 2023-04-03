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
client = MongoClient('mongodb://{}:{}/'.format(config.MONGODB_HOST,
                                               config.MONGODB_PORT),
                     username=config.MONGODB_USERNAME,
                     password=config.MONGODB_PASSWORD,
                     socketTimeoutMS=0, connectTimeoutMS=0)
db = client.main_db
fs = GridFS(db)


# region Useful functions


def _get_one_by_filter(collection, filter, query=None):
    """
    Return request with a document matching filter
    """
    element = collection.find_one(filter, query)
    if element:
        element['_id'] = str(element['_id'])
        return make_response(jsonify(element), 200)
    else:
        return make_response("No element found matching {} in collection {}".format(filter, collection), 403)


def _get_multiple_by_filter(collection, filter, query=None):
    """
    Return request with multiple documents matching filter
    """
    elements = []
    for element in collection.find(filter, query):
        element['_id'] = str(element['_id'])
        elements.append(element)
    return make_response(jsonify(elements), 200)


def _get_file_from_document(document, filename):
    """
    Return request with a file from a document
    """
    if not filename in document:
        return make_response("No file {} found with in document with id {}".format(filename, document['_id']), 403)
    file = fs.get(ObjectId(document[filename]))
    return send_file(file, attachment_filename=file.filename)


def _add_files_to_document(collection, document, files):
    """
    Add files to a document
    """
    # Add files in GridFS
    updates_to_do = {'$set': {}}
    for file_name in files:
        file_id = fs.put(files[file_name], filename=file_name)
        updates_to_do['$set'][file_name] = str(file_id)
    # Add GridFS ids and filenames to document
    collection.update_one({"_id": document["_id"]}, updates_to_do)
    return make_response(jsonify({'success': True}), 200)

# endregion

# region Discovery endpoints


@app.route('/discoveries', methods=['GET'])  # list discoveries given filter
def list_discoveries():
    filter = request.args.get('filter', default=None)
    if filter is not None:
        query = request.args.get('query', default=None)
        return _get_multiple_by_filter(db.discoveries, json.loads(filter), json.loads(query) if query else None)
    else:
        return make_response("You must provide a filter in the request args", 403)


@app.route('/discoveries/<id>', methods=['GET'])  # get a discovery by its id
def get_discovery_by_id(id):
    return _get_one_by_filter(db.discoveries, {"_id": ObjectId(id)})


@app.route('/discoveries/<id>/<file>', methods=['GET'])
def get_discovery_file(id, file):
    """ 
    Get discovery by name 
    """
    discovery = db.discoveries.find_one({"_id": ObjectId(id)})
    if discovery:
        return _get_file_from_document(discovery, file)
    else:
        return make_response("No discovery found with id {}".format(id), 403)


@app.route('/discoveries', methods=['POST'])
def create_discovery():
    """
    Add a new discovery
    """
    added_discovery_id = db.discoveries.insert_one(request.json).inserted_id
    return make_response(jsonify({"ID": str(added_discovery_id)}), 200)


@app.route('/discoveries/<id>/files', methods=['POST'])
def add_discovery_files(id):
    """
    Add files to existing discovery
    """
    discovery = db.discoveries.find_one({"_id": ObjectId(id)})
    if discovery:
        return _add_files_to_document(db.discoveries, discovery, request.files)
    else:
        return make_response("No discovery found with id {}".format(id), 403)


@app.route('/discoveries/<id>', methods=['DELETE'])
def delete_discovery_by_id(id):
    """
    Delete discovery with id
    """
    db.discoveries.delete_one({"_id": ObjectId(id)})
    return make_response(jsonify({'success': True}), 200)


@app.route('/discoveries/', methods=['DELETE'])
def delete_discoveries():
    """
    Delete discoveries associated with a checkpoint_id
    """
    checkpoint_id = int(request.args.get('checkpoint_id', default=None))
    if checkpoint_id is not None:
        db.discoveries.delete_many({"checkpoint_id": checkpoint_id})
        return make_response(jsonify({'success': True}), 200)
    else:
        return make_response("You must provide a checkpoint_id in the request args", 403)

# endregion

# region Checkpointing endpoints


@app.route('/checkpoint_saves', methods=['GET'])
def list_checkpoint_saves():
    """
    List checkpoint saves given filter
    """
    filter = request.args.get('filter', default=None)
    if filter is not None:
        query = request.args.get('query', default=None)
        return _get_multiple_by_filter(db.checkpoint_saves, json.loads(filter),
                                       json.loads(query) if query else None)
    else:
        return make_response("You must provide a "
                             "filter in the request args", 403)


@app.route('/checkpoint_saves/<id>', methods=['GET'])
def get_checkpoint_save_by_id(id):
    return _get_one_by_filter(db.checkpoint_saves, {"_id": ObjectId(id)})


@app.route('/checkpoint_saves/<id>/<file>', methods=['GET'])
def get_checkpoint_save_file(id, file):
    """
    Get file from a checkpoint save by its name
    """
    checkpoint_save = db.checkpoint_saves.find_one({"_id": ObjectId(id)})
    if checkpoint_save:
        return _get_file_from_document(checkpoint_save, file)
    else:
        return make_response(f"No checkpoint_save found with id {id}", 403)


@app.route('/checkpoint_saves', methods=['POST'])  # add a checkpoint save
def create_checkpoint_save():
    """
    Create a checkpoint save
    """
    added_checkpoint_save_id = db.checkpoint_saves.insert_one(
        request.json).inserted_id
    return make_response(jsonify({"ID": str(added_checkpoint_save_id)}), 200)


@app.route('/checkpoint_saves/<id>', methods=['POST'])
def update_checkpoint_save(id):
    """
    Update checkpoint data
    """
    update_doc = {"$set": request.json}
    update_result = db.checkpoint_saves.update_one(
        filter={"_id": ObjectId(id)}, update=update_doc)
    return make_response(jsonify({"success": True}), 200)


@app.route('/checkpoint_saves/<id>/files', methods=['POST'])
def add_checkpoint_save_files(id):
    """
    Add files to a checkpoint 
    """
    checkpoint_save = db.checkpoint_saves.find_one({"_id": ObjectId(id)})
    if checkpoint_save:
        return _add_files_to_document(db.checkpoint_saves,
                                      checkpoint_save, request.files)
    else:
        return make_response(f"No checkpoint_save found with id {id}", 403)


@app.route('/checkpoint_saves/<id>', methods=['DELETE'])
def delete_checkpoint_save_by_id(id):
    """
    Remove a checkpoint save by its id
    """
    db.checkpoint_saves.delete_one({"_id": ObjectId(id)})
    return make_response(jsonify({'success': True}), 200)


@app.route('/checkpoint_saves', methods=['DELETE'])
def delete_checkpoint_saves():
    """
    Remove multiple checkpoint saves given a checkpoint ID
    """
    checkpoint_id_str = request.args.get('checkpoint_id', default=None)
    if checkpoint_id_str is not None:
        checkpoint_id = int(checkpoint_id_str)
        db.checkpoint_saves.delete_many({"checkpoint_id": checkpoint_id})
        return make_response(jsonify({'success': True}), 200)
    else:
        return make_response("You must provide a "
                             "checkpoint_id in the request args", 403)

# endregion

# region Data saves
# GET


@app.route('/data_saves', methods=['GET'])  # list data saves given filter
def list_data_saves():
    filter = request.args.get('filter', default=None)
    if filter is not None:
        query = request.args.get('query', default=None)
        return _get_multiple_by_filter(db.data_saves, json.loads(filter),
                                       json.loads(query) if query else None)
    else:
        return make_response("You must provide a "
                             "filter in the request args", 403)


@app.route('/data_saves/<id>', methods=['GET'])  # get a data save by its id
def get_data_save_by_id(id):
    return _get_one_by_filter(db.data_saves, {"_id": ObjectId(id)})


# get file from a data save by its name
@app.route('/data_saves/<id>/<file>', methods=['GET'])
def get_data_save_file(id, file):
    data_save = db.data_saves.find_one({"_id": ObjectId(id)})
    if data_save:
        return _get_file_from_document(data_save, file)
    else:
        return make_response("No data_save found with id {}".format(id), 403)

# POST


@app.route('/data_saves', methods=['POST'])  # add a data save
def create_data_save():
    added_data_save_id = db.data_saves.insert_one(request.json).inserted_id
    return make_response(jsonify({"ID": str(added_data_save_id)}), 200)


# add files to a data save
@app.route('/data_saves/<id>/files', methods=['POST'])
def add_data_save_files(id):
    data_save = db.data_saves.find_one({"_id": ObjectId(id)})
    if data_save:
        return _add_files_to_document(db.data_saves, data_save, request.files)
    else:
        return make_response("No data_save found with id {}".format(id), 403)
# PATCH


@app.route('/data_saves/<id>', methods=['PATCH'])  # PATCH data
def patch_data_by_id(id):
    update_data = db.data_saves.find_one_and_update(
        {"_id": ObjectId(id)}, {"$set": request.get_json()})
    return make_response(jsonify({"data updated": str(update_data)}), 200)

# DELETE


# remove a data save by its id
@app.route('/data_saves/<id>', methods=['DELETE'])
def delete_data_save_by_id(id):
    db.data_saves.delete_one({"_id": ObjectId(id)})
    return make_response(jsonify({'success': True}), 200)


# remove multiple data save given a data id
@app.route('/data_saves/', methods=['DELETE'])
def delete_data_saves():
    data_id = int(request.args.get('data_id', default=None))
    if data_id is not None:
        db.data_saves.delete_many({"data_id": data_id})
        return make_response(jsonify({'success': True}), 200)
    else:
        return make_response("You must provide a "
                             "data_id in the request args", 403)

# endregion


if __name__ == '__main__':
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT)
