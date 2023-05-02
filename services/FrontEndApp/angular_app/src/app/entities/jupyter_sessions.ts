export interface JupyterSessions {
  id: string;
  kernel: { [key: string]: number };
  name: string;
  notebook: { [key: string]: number };
  path: string;
  type: string;
}
