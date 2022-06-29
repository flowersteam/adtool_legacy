(function(window) {
    window.env = window.env || {};
  
    // Environment variables
    window["env"]["EXPE_DB_API_PORT"] = "${EXPE_DB_API_PORT}";
    window["env"]["EXPEDB_HOST"] = "${EXPEDB_HOST}";
    window["env"]["APP_DB_API_PORT"] = "${APP_DB_API_PORT}";
    window["env"]["APPDB_HOST"] = "${APPDB_HOST}";
    window["env"]["AUTODISC_SERVER_PORT"] = "${AUTODISC_SERVER_PORT}";
    window["env"]["AUTODISC_SERVER_HOST"] = "${AUTODISC_SERVER_HOST}";
    window["env"]["JUPYTER_PORT"] = "${JUPYTER_PORT}";
    window["env"]["JUPYTER_HOST"] = "${JUPYTER_HOST}";
    window["env"]["debug"] = "${DEBUG}";
  })(this);