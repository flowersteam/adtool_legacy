import { Component, OnInit, Input } from '@angular/core';

import { ExpeDbService } from '../../../services/REST-services/expe-db.service';
import { NumberUtilsService } from '../../../services/number-utils.service'
import { ToasterService } from '../../../services/toaster.service';

@Component({
  selector: 'app-discovery',
  templateUrl: './discovery.component.html',
  styleUrls: ['./discovery.component.scss']
})
export class DiscoveryComponent implements OnInit {

  @Input() experiment?: any;

  currentRunIdx: any = [];           // run_idx value defined by the user. Using to show discovery of this specific run_idx
  allSeedSelect: any = { "seeds": [0] };    // seed value defined by the user to visualise it in discoveries tab, first seed check by default
  allSeed: { [key: string]: any[]; } = {}
  nbDiscoveriesDisplay = 20;        // how many discories we want display simultaneously
  maxDiscoveriesDisplay = 20;
  indexDiscoveriesDisplay: number = 0; //index to define wich subarray of run_idx we want display now
  arrayFilterRunIdx: any = [];       //subarray of run_idx we want see now
  sliderDoubleValue = {
    value: 0,
    highValue: 0,
    options: {
      floor: 0,
      ceil: 0,
      translate: (value: number): string => {
        return (value + 1).toString();
      }
    }
  }

  lastExperimentProgress: number = 0;
  constructor(private expeDbService: ExpeDbService, private toasterService: ToasterService, public numberUtilsService: NumberUtilsService) { }

  ngOnInit(): void {
  }

  ngOnChanges(): void {
    this.refreshDiscoveries();
  }

  setCurrentRunIdx() {
    this.currentRunIdx = [];
    for (let index = this.sliderDoubleValue.value; index <= this.sliderDoubleValue.highValue; index++) {
      this.currentRunIdx.push(index)
    }
  }

  defineWhatWeWantVisualise() {
    this.indexDiscoveriesDisplay = 0
    this.setCurrentRunIdx();
    this.getDiscovery();
  }

  definedFilters(): string {
    let filter = "";
    if (this.experiment) {
      this.arrayFilterRunIdx = [];
      for (let i = 0; i <= this.currentRunIdx.length; i = i + this.nbDiscoveriesDisplay) {
        this.arrayFilterRunIdx.push(this.currentRunIdx.slice(
          this.currentRunIdx.length - (i + this.nbDiscoveriesDisplay) >= 0 ?
            this.currentRunIdx.length - (i + this.nbDiscoveriesDisplay) : 0,
          this.currentRunIdx.length - i)
        );
      }
      for (let i = 0; i < this.arrayFilterRunIdx.length; i++) {
        if (this.arrayFilterRunIdx[i].length == 0) {
          this.arrayFilterRunIdx.splice(i, 1);
        }
        else {
          this.arrayFilterRunIdx[i] = this.arrayFilterRunIdx[i].reverse();
        }
      }
      filter = '{"$and":[{"experiment_id":'
        + this.experiment.id.toString()
        + '}, {"run_idx":{"$in":'
        + JSON.stringify(this.arrayFilterRunIdx[this.indexDiscoveriesDisplay] != undefined ? this.arrayFilterRunIdx[this.indexDiscoveriesDisplay] : [])
        + '}}'
      if (this.allSeedSelect["seeds"].length > 0) {
        filter = filter + ', {"seed":{"$in":'
          + JSON.stringify(this.allSeedSelect["seeds"])
          + '}}'
      }
      filter = filter + ']}'
    }
    return filter;
  }

  setIndexDiscoveries(i: number) {
    this.indexDiscoveriesDisplay = this.indexDiscoveriesDisplay + i;
    if (i == 0) {
      this.indexDiscoveriesDisplay = 0;
    }
    else if (this.indexDiscoveriesDisplay < 0) {
      this.indexDiscoveriesDisplay = this.arrayFilterRunIdx.length - 1;
    }
    else if (this.indexDiscoveriesDisplay >= this.arrayFilterRunIdx.length) {
      this.indexDiscoveriesDisplay = 0;
    }

    this.getDiscovery();
  }

  getDiscovery(): void {
    if (this.experiment) {
      for (let index = 0; index < this.experiment.config.nb_seeds; index++) {
        let video = <HTMLVideoElement><any>document.querySelector("#video_" + index.toString());
        if (video) {
          video.src = "";
        }
      }
      let filter = this.definedFilters()
      this.expeDbService.getDiscovery(filter)
      .subscribe(response => {
        if(response.success) {
          let discoveries = response.data ?? []
          if(discoveries.length > 0){
            for(let discovery of discoveries){
              this.expeDbService.getDiscoveryRenderedOutput(discovery._id)
              .subscribe(response => {
                if(response.success) {
                  let video = <HTMLVideoElement> <any> document.querySelector("#video_"+discovery.seed.toString()+"_"+discovery.run_idx.toString());
                  if(video){
                    video.src = window.URL.createObjectURL(response.data);
                  } 
                }
              });
            }
          }
        }
        else{
          this.toasterService.showError(response.message ?? '', "Error getting discoveries");
        }
      });
    }
  }

  refreshDiscoveries() {
    if (this.experiment) {
      this.allSeed = { "seeds": this.numberUtilsService.nFirstIntegers(this.experiment.config.nb_seeds) };
      this.allSeed["seeds"] = this.allSeed["seeds"].filter(x => !new Set(this.allSeedSelect["seeds"]).has(x));
      this.sliderDoubleValue = {
        value: this.sliderDoubleValue.value,
        highValue: this.sliderDoubleValue.highValue == this.lastExperimentProgress && this.experiment.progress > 0 ? this.experiment.progress - 1 : this.sliderDoubleValue.highValue,
        options: {
          floor: 0,
          ceil: this.experiment && this.experiment.progress > 0 ? this.experiment.progress - 1 : this.sliderDoubleValue.options.ceil,
          translate: (value: number): string => {
            return (value + 1).toString();
          }
        }
      }

      this.nbDiscoveriesDisplay > this.lastExperimentProgress
        ? (this.experiment.progress > 0 ? this.nbDiscoveriesDisplay = this.experiment.progress : this.nbDiscoveriesDisplay = 1)
        : this.nbDiscoveriesDisplay = this.nbDiscoveriesDisplay;
      if (this.nbDiscoveriesDisplay > this.maxDiscoveriesDisplay) {
        this.nbDiscoveriesDisplay = this.maxDiscoveriesDisplay
      }

      this.experiment.progress > 0 ? this.lastExperimentProgress = this.experiment.progress - 1 : this.lastExperimentProgress = 0;

      this.defineWhatWeWantVisualise();

    }
  }
}
