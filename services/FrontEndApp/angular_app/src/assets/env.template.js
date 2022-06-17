(function(window) {
    window.env = window.env || {};
  
    // Environment variables
    window["env"]["EXPE_DB_API_PORT"] = "${EXPE_DB_API_PORT}";
    window["env"]["APP_DB_API_PORT"] = "${APP_DB_API_PORT}";
    window["env"]["AUTODISC_SERVER_PORT"] = "${AUTODISC_SERVER_PORT}";
    window["env"]["debug"] = "${DEBUG}";
  })(this);