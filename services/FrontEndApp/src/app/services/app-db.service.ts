import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { catchError, map, tap } from 'rxjs/operators';

import {LightExperiment} from "../entities/light_experiment";

import { Observable, of } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class AppDbService {

  private appDBUrl = "http://127.0.0.1:3000"

  httpOptions = {
    headers: new HttpHeaders({ 'Content-Type': 'application/json' })
  };

  constructor(private http: HttpClient) { }

  /** GET LightExperiments from the AppDB */
  getLightExperiments(): Observable<LightExperiment[]> {
    return this.http.get<LightExperiment[]>(
      this.appDBUrl + "/experiments?select=id,name,created_on,progress,systems(name),explorers(name),input_wrappers(name),output_representations(name)")
      .pipe(
        map(experiments => experiments.map(
          experiment => ({
            ...experiment,
            status: 0 // Remove this once the status is directly accessible in DB
          })
        )),
        catchError(this.handleError<LightExperiment[]>('getLightExperiments', []))
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
