import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { catchError, map } from 'rxjs/operators';

import {
  RESTResponse,
  httpErrorResponseToRESTResponse,
  httpResponseToRESTResponse,
} from '../../entities/rest_response';

import { Observable, of } from 'rxjs';
import { environment } from 'src/environments/environment';

@Injectable({
  providedIn: 'root',
})
export class ExpeDbService {
  private expeDBUrl;

  constructor(private http: HttpClient) {
    this.expeDBUrl =
      'http://' +
      environment.GATEWAY_HOST +
      ':' +
      environment.GATEWAY_PORT +
      '/expe-db-api';
  }

  deleteCheckpointDiscoveries(id: number): Observable<RESTResponse<any>> {
    return this.http
      .delete(this.expeDBUrl + '/discoveries?checkpoint_id=' + id, {
        headers: new HttpHeaders({ 'Content-Type': 'application/json' }),
        observe: 'response',
      })
      .pipe(
        map((response) => {
          return httpResponseToRESTResponse<any>(response);
        }),
        catchError((response) => {
          return of(httpErrorResponseToRESTResponse<any>(response));
        })
      );
  }

  deleteCheckpointSaves(id: number): Observable<RESTResponse<any>> {
    return this.http
      .delete(this.expeDBUrl + '/checkpoint_saves?checkpoint_id=' + id, {
        headers: new HttpHeaders({ 'Content-Type': 'application/json' }),
        observe: 'response',
      })
      .pipe(
        map((response) => {
          return httpResponseToRESTResponse<any>(response);
        }),
        catchError((response) => {
          return of(httpErrorResponseToRESTResponse<any>(response));
        })
      );
  }

  getDiscovery(filter: string): Observable<RESTResponse<string>> {
    return this.http
      .get(
        // temporary fix for excluding mal-formed JSON from the response
        encodeURI(
          this.expeDBUrl +
            '/discoveries?filter=' +
            filter +
            '&query={"output" : false, "raw_output" : false, "params" : false}'
        ),
        {
          headers: new HttpHeaders({ 'Content-Type': 'application/json' }),
          observe: 'response',
          responseType: 'text',
        }
      )
      .pipe(
        map((response) => {
          return httpResponseToRESTResponse<string>(response);
        }),
        catchError((response) => {
          return of(httpErrorResponseToRESTResponse<string>(response));
        })
      );
  }

  getDiscoveryRenderedOutput(id: string): Observable<RESTResponse<Blob>> {
    return this.http
      .get<Blob>(this.expeDBUrl + '/discoveries/' + id + '/rendered_output', {
        headers: new HttpHeaders({ 'Content-Type': 'application/json' }),
        responseType: 'blob' as 'json',
        observe: 'response',
      })
      .pipe(
        map((response) => {
          return httpResponseToRESTResponse<Blob>(response);
        }),
        catchError((response) => {
          return of(httpErrorResponseToRESTResponse<Blob>(response));
        })
      );
  }

  //   public getVideo(url:string): Observable<any> {
  //     const headers = new HttpHeaders({ 'Authorization': 'Bearer ' + this.authenticationService.token, 'Content-Type': 'video/mp4' });
  //     const options = { headers: headers };
  //     return this.http.get(url, options);
  // }
}
