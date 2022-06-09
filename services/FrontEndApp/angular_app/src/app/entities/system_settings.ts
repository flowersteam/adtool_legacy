export interface SystemSettings {
    name: string;
    config: { [key: string]: number };
    input_space: { [key: string]: number };
    output_space: { [key: string]: number };
    step_output_space: { [key: string]: number };
  }