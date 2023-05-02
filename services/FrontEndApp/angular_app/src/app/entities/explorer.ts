export interface Explorer {
  id: number;
  name: string;
  experiment_id: number;
  config: { [key: string]: number };
}
