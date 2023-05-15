import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { catchError, map } from 'rxjs/operators';
import { Experiment } from 'src/app/entities/experiment';

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

  getDiscoveryForExperiment(
    experiment: Experiment
  ): Observable<RESTResponse<string>> {
    // extract necessary data from experiment
    const id = experiment.id;
    const nb_iterations = experiment.config.nb_iterations;
    const nb_seeds = experiment.config.nb_seeds;

    // create ranges for iteration and seed
    const iterations_array = [...Array(nb_iterations).keys()];
    const seeds_array = [...Array(nb_seeds).keys()];

    // construct filter string
    let filter = '{';
    filter +=
      '"$and":[{"experiment_id":' +
      id.toString() +
      '}, {"run_idx":{"$in":' +
      JSON.stringify(iterations_array) +
      '}}';
    if (nb_seeds > 1) {
      filter += ', {"seed":{"$in":' + JSON.stringify(seeds_array) + '}}';
    }
    filter += ']}';

    // call lower-level API
    return this.getDiscovery(filter);
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
