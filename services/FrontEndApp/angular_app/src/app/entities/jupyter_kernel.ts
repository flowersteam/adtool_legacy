export interface JupyterKernel {
  id: string;
  name: string;
  last_activity: string;
  execution_state: string;
  connections: number;
}
