(function (window) {
  window.env = window.env || {};

  // Environment variables
  window["env"]["GATEWAY_PORT"] = "${GATEWAY_PORT}";
  window["env"]["GATEWAY_HOST"] = "${GATEWAY_HOST}";
  window["env"]["debug"] = "${DEBUG}";
})(this);