import { Component, Input, OnInit } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
import { Experiment } from 'src/app/entities/experiment';
import { Media } from 'src/app/entities/media';
import { Observable, Subject } from 'rxjs';
import { map, take, tap } from 'rxjs/operators';

interface MediaSrc {
  seed: number;
  iteration: number;
  src: string;
}

@Component({
  selector: 'app-discovery-vis',
  templateUrl: './discovery-vis.component.html',
  styleUrls: ['./discovery-vis.component.scss'],
})
export class DiscoveryVisComponent implements OnInit {
  @Input() experiment?: Experiment;
  @Input() mediaArray$?: Observable<Media[]>;
  @Input() rawSeedView?: number[];
  @Input() iterationView?: number[];

  seedView?: number[];

  mediaSrcArray$?: Observable<
    { seed: number; iteration: number; src: string }[]
  >;
  mediaSrcArrayFiltered$: Subject<
    { seed: number; iteration: number; src: string }[]
  > = new Subject();

  // life cycle hook for when user changes the view
  ngOnChanges(): void {
    const nb_seeds = this.experiment?.config?.nb_seeds ?? 0;
    const nb_iterations = this.experiment?.config?.nb_iterations ?? 0;

    // null guards
    this.rawSeedView = this.rawSeedView ?? Array(nb_seeds);
    this.iterationView = this.iterationView ?? Array(nb_iterations);

    // initialize view into the discovery data
    this.seedView =
      this.rawSeedView.length > 0 ? this.rawSeedView : Array(nb_seeds);

    // generate filter
    let predicate = (el: MediaSrc) =>
      this.isFiltered(el, this.seedView!, this.iterationView!);

    // apply filter and push to mediaSrcArrayFiltered
    this.mediaSrcArray$
      ?.pipe(
        take(1),
        map((mediaSrcArray) => mediaSrcArray.filter(predicate))
      )
      .subscribe({
        // I doubt that you have to do this manually, but perhaps it is
        // necessary to work well with the async pipe in the template
        next: (mediaSrcArray) => {
          this.mediaSrcArrayFiltered$.next(mediaSrcArray);
        },
        error: (err) => {
          console.log('Error updating filtered discovery.');
        },
        // for debugging only
        complete: () => {
          console.log('Filtering complete.');
        },
      });
  }

  ngOnInit(): void {
    // transform media's src member into the content url
    this.mediaSrcArray$ = this.mediaArray$?.pipe(
      map((mediaArray) => {
        return mediaArray.map(this.generateSrc);
      })
    );
  }

  private isFiltered(
    mediaSrc: MediaSrc,
    seedView: number[],
    iterationView: number[]
  ): boolean {
    return (
      seedView.includes(mediaSrc.seed) &&
      iterationView.includes(mediaSrc.iteration)
    );
  }

  private generateSrc(media: Media): MediaSrc {
    return {
      seed: media.seed,
      iteration: media.iteration,
      src: URL.createObjectURL(media.content),
    };
  }
}
