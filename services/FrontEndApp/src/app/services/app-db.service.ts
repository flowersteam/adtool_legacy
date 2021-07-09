import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { catchError } from 'rxjs/operators';

import {Experiment} from "../entities/experiment";

import { Observable, of } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class AppDbService {

  private appDBUrl = "http://127.0.0.1:3000"

  constructor(private http: HttpClient) { }

  /** GET LightExperiments from the AppDB */
  getLightExperiments(): Observable<Experiment[]> {
    let httpOptions = {
      headers: new HttpHeaders({ 'Content-Type': 'application/json' })
    };

    return this.http.get<Experiment[]>(
      this.appDBUrl + 
      "/experiments?select=id,name,created_on,progress,exp_status," + 
      "systems(name)," +
      "explorers(name)," +
      "input_wrappers(name)," +
      "output_representations(name)",
      httpOptions)
      .pipe(
        catchError(this.handleError<Experiment[]>('getLightExperiments', []))
      );
  }

  getExperimentById(id: number): Observable<Experiment> {
    let httpOptions = {
      headers: new HttpHeaders({ 
        'Content-Type': 'application/json',
        'Accept': 'application/vnd.pgrst.object+json' // Get a single json element instead of an array
      })
    };
    return this.http.get<Experiment>(
      this.appDBUrl + 
      "/experiments?select=id,name,created_on,progress,exp_status,config," + 
      "systems(name,config)," +
      "explorers(name,config)," +
      "input_wrappers(name,config,index)," +
      "output_representations(name,config,index)," +
      "checkpoints(parent_id,status,error_message)" +
      "&id=eq." + id,
      httpOptions)
      .pipe(
        catchError(this.handleError<Experiment>('getExperiment', undefined))
      );
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
