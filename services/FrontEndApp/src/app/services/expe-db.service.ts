import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { catchError } from 'rxjs/operators';

import {Experiment} from "../entities/experiment";

import { Observable, of } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ExpeDbService {

  private expeDBUrl = "http://127.0.0.1:5001"

  constructor(private http: HttpClient) { }

  deleteCheckpointDiscoveries(id: number) {
    return this.http.delete(
      this.expeDBUrl + 
      "/discoveries?checkpoint_id=" + id)
      .pipe(
        catchError(this.handleError<any>('deleteCheckpointDiscoveries', undefined))
      );
  }

  deleteCheckpointSaves(id: number){
    return this.http.delete(
      this.expeDBUrl + 
      "/checkpoint_saves?checkpoint_id=" + id)
      .pipe(
        catchError(this.handleError<any>('deleteCheckpointSaves', undefined))
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
