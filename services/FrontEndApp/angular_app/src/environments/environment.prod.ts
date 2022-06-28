export const environment = {
  production: true,
  EXPE_DB_API_PORT: (window as any)["env"]["EXPE_DB_API_PORT"],
  EXPEDB_HOST: (window as any)["env"]["EXPEDB_HOST"],
  APP_DB_API_PORT: (window as any)["env"]["APP_DB_API_PORT"],
  APPDB_HOST: (window as any)["env"]["APPDB_HOST"],
  AUTODISC_SERVER_PORT: (window as any)["env"]["AUTODISC_SERVER_PORT"],
  AUTODISC_SERVER_HOST: (window as any)["env"]["AUTODISC_SERVER_HOST"],
  JUPYTER_PORT: (window as any)["env"]["JUPYTER_PORT"],
  JUPYTER_HOST: (window as any)["env"]["JUPYTER_HOST"],
};
