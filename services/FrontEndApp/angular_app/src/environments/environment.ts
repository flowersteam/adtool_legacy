// This file can be replaced during build by using the `fileReplacements` array.
// `ng build` replaces `environment.ts` with `environment.prod.ts`.
// The list of file replacements can be found in `angular.json`.

export const environment = {
  production: false,
  EXPE_DB_API_PORT: (window as any)["env"]["EXPE_DB_API_PORT"],
  EXPEDB_HOST: (window as any)["env"]["EXPEDB_HOST"],
  APP_DB_API_PORT: (window as any)["env"]["APP_DB_API_PORT"],
  APPDB_HOST: (window as any)["env"]["APPDB_HOST"],
  AUTODISC_SERVER_PORT: (window as any)["env"]["AUTODISC_SERVER_PORT"],
  AUTODISC_SERVER_HOST: (window as any)["env"]["AUTODISC_SERVER_HOST"],
  JUPYTER_PORT: (window as any)["env"]["JUPYTER_PORT"],
  JUPYTER_HOST: (window as any)["env"]["JUPYTER_HOST"],
  debug: (window as any)["env"]["debug"] || false
};

/*
 * For easier debugging in development mode, you can import the following file
 * to ignore zone related error stack frames such as `zone.run`, `zoneDelegate.invokeTask`.
 *
 * This import should be commented out in production mode because it will have a negative impact
 * on performance if an error is thrown.
 */
// import 'zone.js/plugins/zone-error';  // Included with Angular CLI.
