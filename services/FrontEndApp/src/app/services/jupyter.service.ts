import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { catchError } from 'rxjs/operators';

import { JupyterSessions } from '../entities/jupyter_sessions';
import { JupyterKernel } from '../entities/jupyter_kernel';
import { JupyterDir } from '../entities/jupyter_dir';
import { webSocket, WebSocketSubject } from "rxjs/webSocket";

import { AppDbService } from '../services/app-db.service';
import { Observable, of, Subject } from 'rxjs';
import { v4 as uuid } from 'uuid';

@Injectable({
  providedIn: 'root'
})

export class JupyterService {
  public jupyterUrl = "http://127.0.0.1:8888";
  public jupyWebSocket:any;

  public discoveries:any;
  
  public msg_id_send:string | undefined;
  // public jupy_value_update_number:number = 0;
  public new_data_available = new Subject<boolean>();
  public kernel_restart = new Subject<boolean>();
  public path:any;

  httpOptions = {
    headers: new HttpHeaders({

      'Content-Type': 'application/json' })
  };

  constructor(private http: HttpClient, private appDBService: AppDbService) { }

  createKernel(): Observable<any>{
    return this.http.post<any>(this.jupyterUrl + "/api/kernels", {})
  }

  destroyKernel(kernel_id: string): Observable<any>{
    return this.http.delete<any>(this.jupyterUrl + "/api/kernels/"+kernel_id);
  }

  
  openKernelChannel(kernel_id:string):Observable<any>{
    this.jupyWebSocket = webSocket('ws://localhost:8888/api/kernels/'+kernel_id+'/channels');
    this.jupyWebSocket.subscribe(
      (msg: any) => {
                     
                     if(msg.parent_header.msg_id == this.msg_id_send && msg.content.status == 'ok'){
                      // server inform we can use new data
                      this.new_data_available.next(true);
                     }
                     else if(msg.parent_header.msg_id != this.msg_id_send && msg.parent_header.msg_type == 'execute_request'){
                      // the user clic execute button in notebook so we consider he know the available data
                      this.new_data_available.next(false);
                     }
                    //  else if(msg.parent_header.msg_type == 'shutdown_request' && msg.content.status == 'ok') {
                    //    if(msg.content.restart){
                    //     // the kernel be restart, inform the component to connect it with the new kernel
                    //     this.kernel_restart.next(true);
                    //    }
                    //  }
                     else{
                      //  console.log(msg)
                     }
                    }, // Called whenever there is a message from the server.
      (err: any) => console.log("error append with jupyter kernel server: " + err), // Called if at any point WebSocket API signals some kind of error.
      () => console.log('complete') // Called when connection is closed (for whatever reason).
    );
    return this.jupyWebSocket
  }

  sendToKernel(message:string): Observable<any>{
    let formated_message = this.format_request(message)
    this.msg_id_send = formated_message.parent_header.msg_id
    let response = this.jupyWebSocket.next(formated_message);
    return(response)
  }

  closeKernelChannel(){
    this.jupyWebSocket.complete(); // Closes the connection.
  }

  format_request(code:string){
    let msg_type = 'execute_request'
    let content = { 'code' : code, 'silent':false }
    let hdr = { 'msg_id' : uuid(),
        'username': 'test', 
        'session': uuid(), 
        'data': new Date(),
        'msg_type': msg_type,
        'version' : '5.0' }
    let msg = { 'header': hdr, 'parent_header': hdr, 
        'metadata': {},
        'content': content }
    return msg
  }


  createNotebookDir(experiment_name: string, experiment_id: number, path_template_folder:string):Observable<any>{    
    let response:Observable<any>;
    response = of(this.createJupyDir(experiment_name, experiment_id).subscribe(res=>{
      return this.getAllNoetbooksInDir(path_template_folder).subscribe(res => {
        return this.copyPastAllNotebook(experiment_name, experiment_id, res.content)
      });
    }))
    
    return response;
  }

  createJupyDir(experiment_name: string, experiment_id: number):Observable<any>{
    let response = this.http.put<any>(
      this.jupyterUrl + '/api/contents/Experiments/'+experiment_name+'_'+experiment_id.toString(),
      {
        "name": experiment_name+'_'+experiment_id.toString(),
        "path": experiment_name+'_'+experiment_id.toString(),
        "type": "directory"
      }, 
      this.httpOptions)
    return response;
  }

  getAllNoetbooksInDir(folder_path: string):Observable<JupyterDir>{
    let response = this.http.get<JupyterDir>(
      this.jupyterUrl + '/api/contents/'+folder_path,
      this.httpOptions)
    return response;
  }

  copyPastAllNotebook(experiment_name: string, experiment_id: number, notebooks_dir_content:any):Observable<any>{
    let response = new Observable<any>();
    for(let notebook of notebooks_dir_content){
      console.log(notebook);
      response = this.copyPastNotebook(experiment_name, experiment_id, notebook.path);
      }
    return(response);
  }

  copyPastNotebook(experiment_name: string, experiment_id: number, notebook_path:string):Observable<any>{
    return of(this.http.post<any>(
      this.jupyterUrl + '/api/contents/Experiments/'+experiment_name+'_'+experiment_id.toString(),
      {
        "copy_from" : notebook_path
      },
      this.httpOptions).subscribe())
  }



  /**
   * Handle Http operation that failed.
   * Let the app continue.
   * @param operation - name of the operation that failed
   * @param result - optional value to return as the observable result
  */
   private handleError<T>(operation = 'operation', result?: T) {
    return (error: any): Observable<T> => {

      // TODO: send the error to remote logging infrastructure
      console.error(error); // log to console instead

      // TODO: better job of transforming error for user consumption
      console.log(`${operation} failed: ${error.message}`);

      // Let the app keep running by returning an empty result.
      return of(result as T);
    };
  }

}
